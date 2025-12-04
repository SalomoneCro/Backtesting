import pandas as pd
import numpy as np

# Cargar el archivo CSV
df = pd.read_csv('dailyorder_transactions_rafaga.csv', sep='\t')

# Filtrar solo las operaciones cerradas (CLOSE)
trades = df[df['Status'] == 'CLOSE'].copy()

# Verificar que tenemos datos
if len(trades) == 0:
    print("❌ No se encontraron trades cerrados en el archivo")
    exit()

print("\n" + "="*60)
print("📊 ANÁLISIS DE BACKTESTING - DAILYORDER EA")
print("="*60)

# ============================================
# MÉTRICA 1: Rendimiento Total
# ============================================
initial_balance = 10000.0
final_balance = trades.iloc[-1]['Balance']
total_return = final_balance - initial_balance
total_return_pct = (total_return / initial_balance) * 100

print(f"\n1️⃣  RENDIMIENTO TOTAL")
print(f"   Capital Inicial: ${initial_balance:,.2f}")
print(f"   Capital Final: ${final_balance:,.2f}")
print(f"   Ganancia/Pérdida: ${total_return:,.2f} ({total_return_pct:+.2f}%)")

# ============================================
# MÉTRICA 2: Total de Trades
# ============================================
total_trades = len(trades)
winning_trades = len(trades[trades['Profit'] > 0])
losing_trades = len(trades[trades['Profit'] < 0])
breakeven_trades = len(trades[trades['Profit'] == 0])

print(f"\n2️⃣  DISTRIBUCIÓN DE OPERACIONES")
print(f"   Total de Trades: {total_trades}")
print(f"   Trades Ganadores: {winning_trades} ({winning_trades/total_trades*100:.1f}%)")
print(f"   Trades Perdedores: {losing_trades} ({losing_trades/total_trades*100:.1f}%)")
print(f"   Trades en Breakeven: {breakeven_trades}")

# ============================================
# MÉTRICA 3: Win Rate
# ============================================
win_rate = (winning_trades / total_trades) * 100

print(f"\n3️⃣  WIN RATE (Tasa de Acierto)")
print(f"   {win_rate:.2f}%")

# ============================================
# MÉTRICA 4: Profit Factor
# ============================================
gross_profit = trades[trades['Profit'] > 0]['Profit'].sum()
gross_loss = abs(trades[trades['Profit'] < 0]['Profit'].sum())
profit_factor = gross_profit / gross_loss if gross_loss != 0 else 0

print(f"\n4️⃣  PROFIT FACTOR")
print(f"   {profit_factor:.3f}")
print(f"   (Ganancia Bruta: ${gross_profit:,.2f} / Pérdida Bruta: ${gross_loss:,.2f})")

# ============================================
# MÉTRICA 5: Average Win vs Average Loss
# ============================================
avg_win = trades[trades['Profit'] > 0]['Profit'].mean() if winning_trades > 0 else 0
avg_loss = trades[trades['Profit'] < 0]['Profit'].mean() if losing_trades > 0 else 0
reward_risk_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0

print(f"\n5️⃣  PROMEDIO GANANCIA vs PÉRDIDA")
print(f"   Ganancia Promedio: ${avg_win:,.2f}")
print(f"   Pérdida Promedio: ${avg_loss:,.2f}")
print(f"   Ratio Reward/Risk: {reward_risk_ratio:.2f}")

# ============================================
# MÉTRICA 6: Maximum Drawdown
# ============================================
# Calcular el balance acumulado en cada trade
balance_curve = trades['Balance'].values
running_max = np.maximum.accumulate(balance_curve)
drawdown = balance_curve - running_max
max_drawdown = abs(drawdown.min())
max_drawdown_pct = (max_drawdown / initial_balance) * 100

print(f"\n6️⃣  MAXIMUM DRAWDOWN")
print(f"   ${max_drawdown:,.2f} ({max_drawdown_pct:.2f}%)")

# ============================================
# MÉTRICA 7: Expectativa Matemática
# ============================================
expectancy = (win_rate/100 * avg_win) + ((1 - win_rate/100) * avg_loss)

print(f"\n7️⃣  EXPECTATIVA MATEMÁTICA")
print(f"   ${expectancy:,.2f} por trade")
print(f"   (Ganancia esperada promedio por operación)")

# ============================================
# MÉTRICA 8: Largest Win & Largest Loss
# ============================================
largest_win = trades['Profit'].max()
largest_loss = trades['Profit'].min()

print(f"\n8️⃣  MEJOR Y PEOR TRADE")
print(f"   Mejor Trade: ${largest_win:,.2f}")
print(f"   Peor Trade: ${largest_loss:,.2f}")

# ============================================
# MÉTRICA 9: Comisiones Totales
# ============================================
total_commission = abs(trades['Commission'].sum())
commission_impact_pct = (total_commission / initial_balance) * 100

print(f"\n9️⃣  COMISIONES TOTALES")
print(f"   ${total_commission:,.2f} ({commission_impact_pct:.2f}% del capital inicial)")

# ============================================
# MÉTRICA 10: Sharpe Ratio Simplificado
# ============================================
# Usamos los retornos de cada trade
returns = trades['Profit'].values
avg_return = returns.mean()
std_return = returns.std()
sharpe_ratio = (avg_return / std_return) if std_return != 0 else 0

print(f"\n🔟  SHARPE RATIO (Simplificado)")
print(f"   {sharpe_ratio:.3f}")
print(f"   (Mide retorno ajustado por riesgo. >1 es bueno, >2 es excelente)")

# ============================================
# RESUMEN FINAL
# ============================================
print("\n" + "="*60)
print("📋 RESUMEN EJECUTIVO")
print("="*60)

if total_return > 0:
    resultado = "✅ GANADORA"
elif total_return < 0:
    resultado = "❌ PERDEDORA"
else:
    resultado = "⚖️  BREAKEVEN"

print(f"\nEstrategia: {resultado}")
print(f"Rentabilidad: {total_return_pct:+.2f}%")
print(f"Win Rate: {win_rate:.1f}%")
print(f"Profit Factor: {profit_factor:.2f}")
print(f"Max Drawdown: {max_drawdown_pct:.2f}%")

# Evaluación simple
print("\n🎯 EVALUACIÓN:")
if profit_factor > 1.5 and win_rate > 50 and max_drawdown_pct < 20:
    print("   ✅ Estrategia promisoria - Resultados sólidos")
elif profit_factor > 1.0 and win_rate > 40:
    print("   ⚠️  Estrategia marginal - Necesita optimización")
else:
    print("   ❌ Estrategia no rentable - Requiere revisión profunda")

print("\n" + "="*60 + "\n")