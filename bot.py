import os
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

# --- 1. CONEXIÓN CON LAS LLAVES ---
api_key = os.getenv('ALPACA_API_KEY')
secret_key = os.getenv('ALPACA_SECRET_KEY')

if not api_key or not secret_key:
    print("ERROR: No encuentro las llaves en la caja fuerte de GitHub.")
    exit(1)

cliente_trading = TradingClient(api_key, secret_key, paper=True)
cliente_datos = StockHistoricalDataClient(api_key, secret_key)

# --- 2. ESTADO DE LA CAJA ---
try:
    cuenta = cliente_trading.get_account()
    saldo = float(cuenta.buying_power)
    posiciones = [p.symbol for p in cliente_trading.get_all_positions()]
    print(f"Caja: {saldo:.2f} $ | Cartera: {posiciones}")
except Exception as e:
    print(f"Error al conectar con Alpaca: {e}")
    exit(1)

# --- 3. ESCÁNER DE OPORTUNIDADES ---
print("Buscando empresas en el S&P 500...")
url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
header = {'User-Agent': 'Mozilla/5.0'}
res = requests.get(url, headers=header).text
tabla = pd.read_html(res)[0]
simbolos = [s for s in tabla['Symbol'].tolist() if '.' not in s and '-' not in s]

print(f"Analizando volumen de {len(simbolos)} empresas...")
params = StockBarsRequest(
    symbol_or_symbols=simbolos,
    timeframe=TimeFrame.Day,
    start=datetime.now() - timedelta(days=40),
    end=datetime.now(),
    feed=DataFeed.IEX 
)

velas = cliente_datos.get_stock_bars(params)
df_mercado = velas.df

if df_mercado.empty:
    print("Hoy no hay datos frescos en el mercado. Seguramente esté cerrado.")
    exit(0)

# Buscamos el volumen anormal
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

top_20 = pd.DataFrame(calientes).sort_values(by='Ratio', ascending=False).head(20)['Activo'].tolist()
print(f"Foco del dinero hoy en: {top_20}")

# --- 4. OPERATIVA ---
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
                print(f"COMPRA: {activo} ({cant} acciones)")
        elif pred == 0 and activo in posiciones:
            cliente_trading.close_position(activo)
            print(f"VENTA: {activo}")
    except: continue

print("Operativa finalizada.")
