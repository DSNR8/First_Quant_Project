# Quant project for Mathimatical Modenling and maybe something else 
import numpy as np
import matplotlib.pyplot as plt
import seaborn
import pandas as pd
import yfinance as yf
import scipy.optimize as sco

# Δυναμική Βελτιστοποίηση Χαρτοφυλακίου και Stress Testing με Στοχαστικά Μοντέλα

# Ορίζουμε τα tickers. 
# Για το VUAA διαλέγουμε το γερμανικό ανταλλακτήριο (.DE) και για το AETF την Αθήνα (.AT)
tickers = ['VUAA.DE', 'AETF.AT', 'GLD', 'TLT']

# Κατεβάζουμε τα δεδομένα για τα τελευταία 5 χρόνια
print("Κατέβασμα δεδομένων...")
data = yf.download(tickers, start="2020-01-01")['Close']

# Καθαρίζουμε τα κενά (π.χ. μέρες που η Αθήνα είχε αργία αλλά η Αμερική όχι)
data = data.dropna()

# Κάνουμε κανονικοποίηση (ξεκινάνε όλα από το 100) για να δούμε συγκριτικά την πορεία τους
normalized_data = (data / data.iloc[0]) * 100

# Ζωγραφίζουμε το διάγραμμα
normalized_data.plot(figsize=(12, 6), title="Συγκριτική Πορεία Χαρτοφυλακίου (Base 100)")
plt.xlabel("Ημερομηνία")
plt.ylabel("Κανονικοποιημένη Τιμή")
plt.grid(True)
plt.show()

# Υπολογισμός των απλών ημερήσιων αποδόσεων
returns = data.pct_change().dropna()
print("\nΠρώτες γραμμές των ημερήσιων αποδόσεων:")
print(returns.head())

# Υποθέτουμε ότι έχεις ήδη τρέξει το data = raw_data.ffill().dropna()


# 1. Υπολογισμός των Ημερήσιων Αποδόσεων (Daily Returns)
returns = data.pct_change().dropna()

# 2. Ετησιοποιημένος Πίνακας Συνδιακύμανσης
# Πολλαπλασιάζουμε με το 252 (οι εργάσιμες μέρες του χρηματιστηρίου σε ένα χρόνο) 
# γιατί τα quant models δουλεύουν πάντα με ετήσια νούμερα.
cov_matrix = returns.cov() * 252

print("\n--- Ετησιοποιημένος Πίνακας Συνδιακύμανσης (Covariance Matrix) ---")
print(cov_matrix)

# 3. Ετησιοποιημένες Μέσες Αποδόσεις (Expected Returns)
mean_returns = returns.mean() * 252
print("\n--- Μέσες Ετήσιες Αποδόσεις ---")
print(mean_returns)

# Υποθέτουμε risk-free rate (π.χ. το επιτόκιο της ΕΚΤ ή κρατικών ομολόγων, ας πούμε 3% ή 0.03)
risk_free_rate = 0.03

# Συνάρτηση που υπολογίζει την απόδοση, το ρίσκο και το Sharpe Ratio ενός χαρτοφυλακίου
def portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate):
    # Ετήσια Απόδοση: w^T * μ
    returns = np.sum(mean_returns * weights) 
    # Ετήσιος Κίνδυνος (Μεταβλητότητα / Standard Deviation): sqrt(w^T * Σ * w)
    std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) 
    return returns, std_dev

# Η συνάρτηση-στόχος (Objective Function) που θέλουμε να ελαχιστοποιήσουμε
def negative_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
    p_ret, p_std = portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate)
    # Επιστρέφουμε το αρνητικό Sharpe
    return - (p_ret - risk_free_rate) / p_std

# Ο αριθμός των assets μας (στην περίπτωσή μας είναι 4: VUAA, AETF, GLD, TLT)
num_assets = len(tickers)

# Βάζουμε ίσα αρχικά βάρη για να ξεκινήσει ο αλγόριθμος να ψάχνει (25% στο καθένα)
initial_weights = num_assets * [1. / num_assets,]

# 1ος Περιορισμός: Το άθροισμα των βαρών πρέπει να είναι 1 (100% των χρημάτων μας)
# Η scipy θέλει τους περιορισμούς σε μορφή λεξικού (dictionary)
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

# 2ος Περιορισμός (Bounds): Κάθε asset μπορεί να έχει βάρος από 0 (0%) έως 1 (100%)
# Αυτό πρακτικά απαγορεύει το short selling.
# Βάζουμε όριο (cap) 35% στο κάθε asset. 
# Έτσι ο αλγόριθμος αναγκάζεται να αγοράσει τουλάχιστον 3 διαφορετικά assets!
bounds = tuple((0, 0.35) for asset in range(num_assets))

print("\nΤρέχουμε τον αλγόριθμο βελτιστοποίησης...")

# ΕΔΩ ΓΙΝΕΤΑΙ Η ΜΑΓΕΙΑ! Κάνουμε minimize το negative sharpe
optimal_portfolio = sco.minimize(negative_sharpe_ratio, initial_weights, 
                                 args=(mean_returns, cov_matrix, risk_free_rate), 
                                 method='SLSQP', bounds=bounds, constraints=constraints)

# Αποθηκεύουμε τα βέλτιστα βάρη που βρήκε ο αλγόριθμος
optimal_weights = optimal_portfolio.x

# Τυπώνουμε τα αποτελέσματα με ωραίο format
print("\n--- ΒΕΛΤΙΣΤΟ ΧΑΡΤΟΦΥΛΑΚΙΟ (Max Sharpe Ratio) ---")
for i in range(num_assets):
    print(f"{tickers[i]}: {optimal_weights[i]*100:.2f}%")

# Υπολογίζουμε τα τελικά metrics του χαρτοφυλακίου μας
opt_ret, opt_std = portfolio_performance(optimal_weights, mean_returns, cov_matrix, risk_free_rate)
print(f"\nΑναμενόμενη Ετήσια Απόδοση: {opt_ret*100:.2f}%")
print(f"Αναμενόμενος Ετήσιος Κίνδυνος (Vol): {opt_std*100:.2f}%")
print(f"Sharpe Ratio: {(opt_ret - risk_free_rate) / opt_std:.2f}")

# --- MONTE CARLO SIMULATION (Γεωμετρική Κίνηση Brown) ---

print("\nΞεκινάει η Προσομοίωση Monte Carlo...")

initial_investment = 10000 # Έστω ότι επενδύουμε 10.000€
days_to_simulate = 252     # 1 χρόνος (εργάσιμες μέρες)
num_simulations = 10000    # Τρέχουμε 10.000 διαφορετικά παράλληλα σύμπαντα!

# Μετατρέπουμε την ετήσια απόδοση/ρίσκο του βέλτιστου χαρτοφυλακίου σε ΗΜΕΡΗΣΙΑ
daily_return = opt_ret / 252
daily_volatility = opt_std / np.sqrt(252)

# Φτιάχνουμε έναν κενό πίνακα για να αποθηκεύσουμε τα "μονοπάτια"
simulated_portfolios = np.zeros((days_to_simulate, num_simulations))
simulated_portfolios[0] = initial_investment

# Η λούπα της Στοχαστικής Ανέλιξης (GBM)
for t in range(1, days_to_simulate):
    # Το Z είναι η τυχαία μεταβλητή (από κανονική κατανομή N(0,1)) που παίζει τον ρόλο του dW_t
    Z = np.random.standard_normal(num_simulations)
    
    # Η εξίσωση της Γεωμετρικής Κίνησης Brown
    simulated_portfolios[t] = simulated_portfolios[t-1] * np.exp(
        (daily_return - 0.5 * daily_volatility**2) + daily_volatility * Z
    )

# Παίρνουμε τις τελικές αξίες του χαρτοφυλακίου (την 252η μέρα) από όλα τα 10.000 σενάρια
final_values = simulated_portfolios[-1]

# Υπολογισμός του Value at Risk (VaR) στο 95%
var_95 = np.percentile(final_values, 5) # Βρίσκουμε το όριο για το χειρότερο 5% των σεναρίων

print(f"\n--- ΑΠΟΤΕΛΕΣΜΑΤΑ MONTE CARLO (1 Έτος) ---")
print(f"Αρχικό Κεφάλαιο: {initial_investment} €")
print(f"Μέση Αναμενόμενη Αξία: {np.mean(final_values):.2f} €")
print(f"Value at Risk (95%): Στο χειρότερο 5% των σεναρίων, η αξία θα πέσει κάτω από {var_95:.2f} €")
print(f"Μέγιστη Αναμενόμενη Ζημιά (με 95% σιγουριά): {initial_investment - var_95:.2f} €")

# Ζωγραφίζουμε τα πρώτα 100 μονοπάτια για να δούμε τον στοχαστικό "θόρυβο"
plt.figure(figsize=(10, 6))
plt.plot(simulated_portfolios[:, :100], color='blue', alpha=0.1)
plt.axhline(initial_investment, color='black', linestyle='--', label='Αρχικό Κεφάλαιο (10.000€)')
plt.title('Monte Carlo: 100 Πιθανά Στοχαστικά Μονοπάτια του Χαρτοφυλακίου')
plt.xlabel('Ημέρες')
plt.ylabel('Αξία Χαρτοφυλακίου (€)')
plt.legend()
plt.grid(True)
plt.show()

# (Προαιρετικά) Ένα ιστόγραμμα για να δούμε την κατανομή των τελικών αποδόσεων
plt.figure(figsize=(10, 6))
plt.hist(final_values, bins=100, color='orange', alpha=0.7)
plt.axvline(var_95, color='red', linestyle='dashed', linewidth=2, label=f'VaR 95%: {var_95:.0f}€')
plt.axvline(initial_investment, color='black', linestyle='solid', linewidth=2, label='Αρχικό Κεφάλαιο')
plt.title('Κατανομή Τελικής Αξίας Χαρτοφυλακίου μετά από 1 Χρόνο')
plt.xlabel('Τελική Αξία (€)')
plt.ylabel('Συχνότητα')
plt.legend()
plt.show()

# --- STRESS TESTING (Σενάριο Κραχ / Black Swan Event) ---

print("\nΞεκινάει το Stress Test (Σενάριο Κραχ)...")

# Πειράζουμε τις παραμέτρους (Shock the parameters)
# Υποθέτουμε μια βίαιη πτώση αγοράς (-20% απόδοση) και διπλασιασμό του πανικού (volatility x 2)
stressed_ret = -0.20  
stressed_std = opt_std * 2.0 

# Μετατροπή σε ημερήσια νούμερα
stressed_daily_return = stressed_ret / 252
stressed_daily_volatility = stressed_std / np.sqrt(252)

# Νέος πίνακας για το Stress Test
stressed_portfolios = np.zeros((days_to_simulate, num_simulations))
stressed_portfolios[0] = initial_investment

# Η λούπα της Στοχαστικής Ανέλιξης με τις stressed παραμέτρους
for t in range(1, days_to_simulate):
    Z = np.random.standard_normal(num_simulations)
    stressed_portfolios[t] = stressed_portfolios[t-1] * np.exp(
        (stressed_daily_return - 0.5 * stressed_daily_volatility**2) + stressed_daily_volatility * Z
    )

stressed_final_values = stressed_portfolios[-1]

# Υπολογισμός του Stressed VaR στο 95%
stressed_var_95 = np.percentile(stressed_final_values, 5)

print(f"\n--- ΑΠΟΤΕΛΕΣΜΑΤΑ STRESS TEST (1 Έτος - Bear Market) ---")
print(f"Αρχικό Κεφάλαιο: {initial_investment} €")
print(f"Μέση Αναμενόμενη Αξία σε Κραχ: {np.mean(stressed_final_values):.2f} €")
print(f"Stressed VaR (95%): Στο χειρότερο 5% των σεναρίων, η αξία θα καταρρεύσει κάτω από τα {stressed_var_95:.2f} €")
print(f"ΠΡΑΓΜΑΤΙΚΗ Μέγιστη Αναμενόμενη Ζημιά: {initial_investment - stressed_var_95:.2f} € (Χάνεις το {((initial_investment - stressed_var_95)/initial_investment)*100:.2f}% του κεφαλαίου σου)")

# Το τελικό γράφημα κατανομής (Το διάγραμμα "εφιάλτης" κάθε Fund Manager)
plt.figure(figsize=(10, 6))
plt.hist(stressed_final_values, bins=100, color='darkred', alpha=0.7)
plt.axvline(stressed_var_95, color='black', linestyle='dashed', linewidth=2, label=f'Stressed VaR 95%: {stressed_var_95:.0f}€')
plt.axvline(initial_investment, color='blue', linestyle='solid', linewidth=2, label='Αρχικό Κεφάλαιο (10.000€)')
plt.title('Stress Test: Κατανομή Τελικής Αξίας σε Σενάριο Ακραίας Κρίσης (Black Swan)')
plt.xlabel('Τελική Αξία (€)')
plt.ylabel('Συχνότητα')
plt.legend()
plt.show()
