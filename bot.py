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

# --- 2. ESTADO DE LA CUENTA ---
try:
    cuenta = cliente_trading.get_account()
    saldo = float(cuenta.buying_power)
    posiciones = [p.symbol for p in cliente_trading.get_all_positions()]
    print(f"Caja disponible: {saldo:.2f} $")
except Exception as e:
    print(f"Error de conexión: {e}")
    exit(1)

# --- 3. BÚSQUEDA DE OPORTUNIDADES ---
try:
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    header = {'User-Agent': 'Mozilla/5.0'}
    res = requests.get(url, headers=header).text
    
    # El truco está aquí: usar io.StringIO para que pandas no se líe
    tabla = pd.read_html(io.StringIO(res))[0]
    simbolos = [s for s in tabla['Symbol'].tolist() if '.' not in s and '-' not in s]
except Exception as e:
    print(f"Error al leer la lista de empresas: {e}")
    exit(1)

print(f"Analizando {len(simbolos)} activos...")

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
except Exception as e:
    print(f"No se han podido descargar datos hoy: {e}")
    exit(0)

# Filtrar volumen anormal
calientes = []
for ticker in simbolos:
    try:
        if ticker in df_mercado.index:
            d = df_mercado.loc[ticker]
            if len(d) > 20:
                media = d['volume'][:-1].mean()
                hoy = d['volume'].iloc[-1]
                if media > 0:
                    calientes.append({'Activo': ticker, 'Ratio': hoy / media})
    except: continue

if not calientes:
    print("No se han detectado movimientos inusuales.")
    exit(0)

top_20 = pd.DataFrame(calientes).sort_values(by='Ratio', ascending=False).head(20)['Activo'].tolist()
print(f"Activos con mayor volumen hoy: {top_20}")

# --- 4. EJECUCIÓN ---
presupuesto = saldo * 0.15
for activo in top_20:
    try:
        datos = df_mercado.loc[activo].copy()
        datos['retorno'] = datos['close'].pct_change()
        datos['obj'] = (datos['retorno'].shift(-1) > 0).astype(int)
        datos['m5'] = datos['close'].rolling(5).mean()
        datos['m20'] = datos['close'].rolling(20).mean()
        datos = datos.dropna()
        
        if len(datos) < 5: continue
        
        X = datos[['m5', 'm20', 'retorno']]
        y = datos['obj']
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X[:-1], y[:-1])
        
        pred = model.predict(X.iloc[-1:])[0]
        precio = datos['close'].iloc[-1]
        
        if pred == 1 and activo not in posiciones:
            cant = math.floor(presupuesto / precio)
            if cant > 0:
                cliente_trading.submit_order(MarketOrderRequest(
                    symbol=activo, qty=cant, side=OrderSide.BUY, time_in_force=TimeInForce.DAY
                ))
                print(f"Comprando: {activo}")
        elif pred == 0 and activo in posiciones:
            cliente_trading.close_position(activo)
            print(f"Vendiendo: {activo}")
    except: continue

print("Proceso finalizado correctamente.")
