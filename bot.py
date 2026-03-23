import os
import io
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed
from datetime import datetime, timedelta
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import math
import requests

# --- 1. CONEXIÓN ---
api_key = os.getenv('ALPACA_API_KEY')
secret_key = os.getenv('ALPACA_SECRET_KEY')

cliente_trading = TradingClient(api_key, secret_key, paper=True)
cliente_datos = StockHistoricalDataClient(api_key, secret_key)

# --- 2. ESTADO DE LA CUENTA (CON FRENO DE MANO) ---
try:
    cuenta = cliente_trading.get_account()
    # Usamos 'cash' en vez de 'buying_power' para evitar el dinero prestado
    dinero_real = float(cuenta.cash)
    posiciones_actuales = {p.symbol: p for p in cliente_trading.get_all_positions()}
    
    # Solo vamos a arriesgar el 80% del total para tener siempre un colchón
    presupuesto_total = dinero_real * 0.80
    
    print(f"Efectivo real en caja: {dinero_real:.2f} $")
    print(f"Presupuesto máximo para hoy: {presupuesto_total:.2f} $")
except Exception as e:
    print(f"Error al mirar la caja: {e}")
    exit(1)

# --- 3. BÚSQUEDA DE OPORTUNIDADES ---
try:
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    header = {'User-Agent': 'Mozilla/5.0'}
    res = requests.get(url, headers=header).text
    tabla = pd.read_html(io.StringIO(res))[0]
    simbolos = [s for s in tabla['Symbol'].tolist() if '.' not in s and '-' not in s]
except Exception as e:
    print(f"Error con la Wikipedia: {e}")
    exit(1)

params = StockBarsRequest(
    symbol_or_symbols=simbolos,
    timeframe=TimeFrame.Day,
    start=datetime.now() - timedelta(days=40),
    end=datetime.now(),
    feed=DataFeed.IEX 
)

try:
    velas = cliente_datos.get_stock_bars(params)
    df_mercado = velas.df
except:
    print("No hay datos de mercado ahora mismo.")
    exit(0)

calientes = []
for ticker in simbolos:
    try:
        if ticker in df_mercado.index:
            d = df_mercado.loc[ticker]
            if len(d) > 20:
                media = d['volume'][:-1].mean()
                hoy = d['volume'].iloc[-1]
                if media > 0 and hoy > media * 1.5:
                    calientes.append({'Activo': ticker, 'Ratio': hoy / media})
    except: continue

if not calientes:
    print("Hoy el mercado está tranquilito, no compro nada.")
    exit(0)

# Cogemos el Top 10 para no dispersar mucho el dinero
top_picks = pd.DataFrame(calientes).sort_values(by='Ratio', ascending=False).head(10)['Activo'].tolist()
print(f"Candidatos de hoy: {top_picks}")

# --- 4. OPERATIVA CONTROLADA ---
# Dividimos el presupuesto total entre el número de candidatos
if len(top_picks) > 0:
    dinero_por_accion = presupuesto_total / len(top_picks)
else:
    dinero_por_accion = 0

for activo in top_picks:
    try:
        datos = df_mercado.loc[activo].copy()
        datos['retorno'] = datos['close'].pct_change()
        datos['obj'] = (datos['retorno'].shift(-1) > 0).astype(int)
        datos['m5'] = datos['close'].rolling(5).mean()
        datos['m20'] = datos['close'].rolling(20).mean()
        datos = datos.dropna()
        
        if len(datos) < 10: continue
        
        X = datos[['m5', 'm20', 'retorno']]
        y = datos['obj']
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X[:-1], y[:-1])
        
        pred = model.predict(X.iloc[-1:])[0]
        precio_actual = datos['close'].iloc[-1]
        
        # Lógica de compra: si la IA dice sí y no la tenemos ya
        if pred == 1 and activo not in posiciones_actuales:
            cantidad = math.floor(dinero_por_accion / precio_actual)
            if cantidad > 0:
                cliente_trading.submit_order(MarketOrderRequest(
                    symbol=activo, qty=cantidad, side=OrderSide.BUY, time_in_force=TimeInForce.DAY
                ))
                print(f"COMPRA: {activo} ({cantidad} acciones a {precio_actual:.2f}$)")
        
        # Lógica de venta: si la IA dice no y la tenemos en cartera
        elif pred == 0 and activo in posiciones_actuales:
            cliente_trading.close_position(activo)
            print(f"VENTA: {activo} por cambio de tendencia.")
            
    except Exception as e:
        continue

print("Jornada terminada. Todo bajo control.")
