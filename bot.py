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
from textblob import TextBlob

# --- 1. CONEXION ---
api_key = os.getenv('ALPACA_API_KEY')
secret_key = os.getenv('ALPACA_SECRET_KEY')

cliente_trading = TradingClient(api_key, secret_key, paper=True)
cliente_datos = StockHistoricalDataClient(api_key, secret_key)

# --- FUNCION LECTORA DE NOTICIAS ---
def obtener_nota_prensa(ticker, key, secret):
    try:
        url = f"https://data.alpaca.markets/v1beta1/news?symbols={ticker}&limit=3"
        headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
        res = requests.get(url, headers=headers).json()
        
        if 'news' not in res or len(res['news']) == 0:
            return 0.0
            
        nota_total = 0
        for articulo in res['news']:
            analisis = TextBlob(articulo['headline'])
            nota_total += analisis.sentiment.polarity
            
        return nota_total / len(res['news'])
    except Exception as e:
        return 0.0

# --- 2. CAJA Y PARACAIDAS (STOP LOSS / TAKE PROFIT) ---
try:
    cuenta = cliente_trading.get_account()
    dinero_real = float(cuenta.cash)
    posiciones = cliente_trading.get_all_positions()
    posiciones_actuales = {p.symbol: p for p in posiciones}
    
    presupuesto_total = dinero_real * 0.80
    
    print(f"Efectivo real en caja: {dinero_real:.2f} $")
    print("Revisando paracaidas de las posiciones abiertas...")
    
    for pos in posiciones:
        ganancia_pct = float(pos.unrealized_plpc)
        simbolo = pos.symbol
        
        # Take Profit: +8%, Stop Loss: -3%
        if ganancia_pct >= 0.08:
            cliente_trading.close_position(simbolo)
            print(f"TAKE PROFIT: Vendiendo {simbolo} con un +{ganancia_pct*100:.2f}% a la saca.")
            del posiciones_actuales[simbolo]
        elif ganancia_pct <= -0.03:
            cliente_trading.close_position(simbolo)
            print(f"STOP LOSS: Cortando sangria en {simbolo} ({ganancia_pct*100:.2f}%).")
            del posiciones_actuales[simbolo]

except Exception as e:
    print(f"Error al mirar la caja o posiciones: {e}")
    exit(1)

# --- 3. BUSQUEDA DE OPORTUNIDADES (HISTORICO DE 2 AÑOS) ---
try:
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    header = {'User-Agent': 'Mozilla/5.0'}
    res = requests.get(url, headers=header).text
    tabla = pd.read_html(io.StringIO(res))[0]
    # Filtramos a 100 empresas para procesar rápido los 2 años de datos
    simbolos = [s for s in tabla['Symbol'].tolist() if '.' not in s and '-' not in s][:100]
except Exception as e:
    print(f"Error con la Wikipedia: {e}")
    exit(1)

params = StockBarsRequest(
    symbol_or_symbols=simbolos,
    timeframe=TimeFrame.Day,
    start=datetime.now() - timedelta(days=730),
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
            if len(d) > 200: # Exigimos histórico sólido
                media = d['volume'][:-1].mean()
                hoy = d['volume'].iloc[-1]
                if media > 0 and hoy > media * 1.5:
                    calientes.append({'Activo': ticker, 'Ratio': hoy / media})
    except: continue

if not calientes:
    print("Hoy el mercado esta tranquilito, no compro nada.")
    exit(0)

top_picks = pd.DataFrame(calientes).sort_values(by='Ratio', ascending=False).head(5)['Activo'].tolist()
print(f"Candidatos calientes de hoy: {top_picks}")

# --- 4. IA AVANZADA (DATOS + NOTICIAS) ---
if len(top_picks) > 0:
    dinero_por_accion = presupuesto_total / len(top_picks)
else:
    dinero_por_accion = 0

for activo in top_picks:
    if activo in posiciones_actuales:
        continue # Si ya la tenemos, saltamos a la siguiente

    try:
        datos = df_mercado.loc[activo].copy()
        datos['retorno'] = datos['close'].pct_change()
        
        datos['obj'] = (datos['retorno'].shift(-1) > 0).astype(int)
        
        datos['m5'] = datos['close'].rolling(5).mean()
        datos['m20'] = datos['close'].rolling(20).mean()
        datos['volatilidad'] = datos['retorno'].rolling(10).std()
        
        delta = datos['close'].diff()
        ganancia = delta.where(delta > 0, 0).rolling(14).mean()
        perdida = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = ganancia / perdida
        datos['rsi'] = 100 - (100 / (1 + rs))
        
        datos = datos.dropna()
        if len(datos) < 200: continue
        
        X = datos[['m5', 'm20', 'retorno', 'volatilidad', 'rsi']]
        y = datos['obj']
        
        # Modelo más robusto con los datos nuevos
        model = RandomForestClassifier(n_estimators=100, max_depth=7, random_state=42)
        model.fit(X[:-1], y[:-1])
        
        pred = model.predict(X.iloc[-1:])[0]
        precio_actual = datos['close'].iloc[-1]
        
        if pred == 1:
            nota_prensa = obtener_nota_prensa(activo, api_key, secret_key)
            
            if nota_prensa < -0.2:
                print(f"FRENO: La IA iba a comprar {activo} pero las noticias son feas ({nota_prensa:.2f}).")
            else:
                cantidad = math.floor(dinero_por_accion / precio_actual)
                if cantidad > 0:
                    cliente_trading.submit_order(MarketOrderRequest(
                        symbol=activo, qty=cantidad, side=OrderSide.BUY, time_in_force=TimeInForce.DAY
                    ))
                    print(f"COMPRA: {activo} ({cantidad} acciones a {precio_actual:.2f}$). Nota prensa: {nota_prensa:.2f}")
            
    except Exception as e:
        continue

print("Jornada terminada. Todo bajo control, jefe.")
