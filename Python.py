stock_prices = [100, 101, 102, 98, 97]
for i in range(1, len(stock_prices)):
    daily_return = (stock_prices[i] - stock_prices[i-1]) / stock_prices[i-1]
    print("For Loop : " + str(daily_return))



principal = 500
rate = 0.07
years = 0
while principal < 1000:
    principal *= (1 + rate)
    years += 1
print("While Loop: " + str(years))



principal=1000
rate=0.05
years=0
while principal<2000:
    principal*= (1+rate)
    years+=1
print(years)



stock_prices=[105, 107, 104, 106, 103]
for i in range(1,len(stock_prices)):
    daily_return=(stock_prices[i]-stock_prices[i-1])/stock_prices[i]
print("Exercice 1 " + str(daily_return))



stock_prices = [105, 107, 104, 106, 103]
daily_returns = []
for i in range(1, len(stock_prices)):
    daily_return = (stock_prices[i] - stock_prices[i - 1]) / stock_prices[i]
    daily_returns.append(daily_return)

average_return = sum(daily_returns) / len(daily_returns)
print("Average return" + str(average_return))



#nombre d'années nécessaire pour avoir P = 1000
principal = 500
rate = 0.07
year = 0 
while principal < 1000 : 
    principal *= (1+rate)
    year += 1 
print("Nombres d'années: " + str(year) + "et le principal est de: " + str(principal))



#Final value of the investment 
principal = 500
rate = 0.07
year = 0
while principal < 1000 : 
    principal *= (1+rate)
    year += 1 
    if principal >= 1000 :
        print(principal)
#test
test = 500*(1.07)**11
print(test)



#conditional statments
pe_ratio = 20
if pe_ratio < 15:
    print("Buy the stock.")
elif pe_ratio > 25:
    print("Sell the stock.")
else:
    print("Hold the stock.")



bond_yield = 4
if bond_yield >= 4 :
    print("Buy the bond")
else :
    print("Don't buy the bond")



pe_ratio = 17
if pe_ratio < 15:
    print("Buy the stock")
elif pe_ratio > 20:
    print("Sell the stock")
else:
    print("Hold the stock")


pe_ratio = 15
if 14 < pe_ratio < 16:
    print("Buy the stock")
elif 23 < pe_ratio < 27:
    print("Sell the stock")
else:
    print("Hold the stock")



#Object oriented programming

class Bond:
    def __init__(self, par_value, coupon_rate, maturity):
        self.par_value = par_value
        self.coupon_rate = coupon_rate
        self.maturity = maturity
    def current_yield(self, market_price):
        return (self.coupon_rate*self.par_value)/market_price
ten_year_note = Bond(1000, 0.025, 10)
yield_on_note = ten_year_note.current_yield(950)
print (yield_on_note)



class Stock:
    def __init__(self, stock_name, current_price, dividend):
        self.stock_name = stock_name
        self.current_price = current_price
        self.dividend = dividend
    def yield_dividend (self):
        return (self.dividend / self.current_price)
APPL = Stock("APPL", 200, 0.75)
print(APPL.yield_dividend())        


class CurrencyConverter:
    def __init__(self):
        self.rates = {}
    def conversion_rate(self, source, target, rate):
        key = (source, target)  #permet de combiner deux devises EX source USD target EUR = key(USD/EUR)
        self.rates[key] = rate 
    def convert(self, amount, source, target):
        key = (source, target)
        if key in self.rates:
            conversion_rate = self.rates[key]
            converted_amount = amount * conversion_rate
            return converted_amount
        else :
            print("Conversion rate not available.")



# Currency converter test
converter = CurrencyConverter()
converter.conversion_rate("USD","EUR", 0.8)
print(converter.rates)
converter.conversion_rate("USD","JPY",140)
print(converter.rates)
USDtoEUR = converter.convert(100,"USD","EUR")
print("100 USD is equal to :", USDtoEUR, "EUR")
USDtoJPY = converter.convert(250,"USD","JPY")
print("250 USD is equal to : ", USDtoJPY,"JPY")


#Mathematical Tools 

import numpy as np

prices = np.array([100, 102, 104, 101, 99, 98])
returns = (prices[1:] - prices[:-1]) / prices[:-1]
print("Daily returns:", returns)

annual_volatility = np.std(returns) * np.sqrt(252)     #% Assuming 252 trading days
print("Annualized volatility:", annual_volatility)


# Exercice 1 : Stock price simulations

np.random.seed(0)
daily_returns = np.random.normal(0.001, 0.02, 1000)   # (moyenne, écart-type, taille)
stock_prices = [100]
for r in daily_returns:                               # r est la variation quotidienne des rendements pour le jour en cours.
    stock_prices.append(stock_prices[-1]*(1+r))
print(stock_prices[-1])


#Exercice 2 : Portfolio variance

import numpy as np
values = np.array([0.1, 0.2])
weights = np.array([0.4, 0.6])
average = np.average(values, weights=weights)
variance = np.average((values-average)**2, weights=weights)
print(variance)


# Exercice 3 : Efficient frontier

# `rng=np.random.default_rng(seed=42)` is creating a random number generator object using the
# `default_rng` function from the `numpy.random` module. The `seed` parameter is set to 42, which
# ensures that the random numbers generated will be the same each time the code is run. This is useful
# for reproducibility, as it allows you to obtain the same random numbers for testing or debugging
# purposes.
rng=np.random.default_rng(seed=42)
weights = rng.random((5, 2))
print("Weights are :",weights)

returns=np.array([0.1, 0.15])
rendement=returns*weights
rendement2 =rendement.sum(axis=1)
#print ("rendement:", rendement)
print ("Returns in function of the weights are :", rendement2)



#Data Visualization Tools 

#Exercise 1: Plotting Stock Prices using Matplotlib

import matplotlib.pyplot as plt
import seaborn as sns
dates = [1,2,3,4,5,6,7,8,9,10]
stock_prices = [105, 103, 106, 109, 108, 107, 110, 112, 111, 113]
#plt.plot(stock_prices)
#plt.title('Stock Prices over 10 days')
#plt.xlabel('Days')
#plt.ylabel('Stock Prices')
#plt.show()
stock_prices2 = [107, 108, 107, 107, 106, 108, 109, 108, 109, 110]
plt.plot(dates,stock_prices, label="stock_prices1", color = "blue")
plt.plot(dates,stock_prices2, label="stock_prices2", color = "orange")
plt.title('Stock Prices 1 & 2 over 10 days')
plt.legend ("Stock1 = blue")
plt.legend("Stock2 = orange")
plt.xlabel('Days')
plt.ylabel('Stock Prices')
plt.show()
	

#Exercise 2: Visualizing Distributions using Seaborn


import matplotlib.pyplot as plt
import seaborn as sns
returns = [0.05, -0.02, 0.03, -0.01, 0.02, 0.03, -0.03, 0.01, 0.04, -0.01]
sns.histplot(returns, bins=5, kde=True)
plt.title(’Distribution of Stock Returns’)
plt.show()


#Financial Times Series

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#1 

dates = ["4th January", "5th January", "6th January"]
stock_prices1 = [155, 156, 153]

def calculate_average(prices):
    return sum(prices)/len(prices)
average_prices = calculate_average(stock_prices1)
print('Average stock prices :', {average_prices})


#2

#on cherche à trouver le jour associé à la valeur la plus haute. 
#Pour cela on utilise la fonction index pour garder en mémoire une valeur.
max_prices_index = stock_prices1.index(max(stock_prices1))
dates_highprice = dates[max_prices_index]
print(dates_highprice)


#3

dates.append("7th January")
dates.append("8th January")
stock_prices1.append(157)
stock_prices1.append(158)

# Idée : calculer la moving average pour avoir la tendance globale sur la timeline
# pourquoi ça ne fonctionne pas c'est la formule du quizz
# moving_average = rolling(window=5).mean()


if not stock_prices1:
    print("Aucune donnée de prix n'est disponible.")
else:
    n = len(dates)
    def moving_average(stock_prices1, n):
        ret = np.cumsum(stock_prices1)
        ret [n:] = ret[n:] - ret[:-n]
        return ret[n-1:]/n
    
mov_avg = moving_average(stock_prices1, 5)

if mov_avg > average_prices:
    print("la tendance est croissante")
elif mov_avg < average_prices :
    print ("la tendance est décroissante")
else :
    print ("la tendance est stable")


#Histogramme

import matplotlib.pyplot as plt
import seaborn as sns
returns = [0.05, -0.02, 0.03, -0.01, 0.02, 0.03, -0.03, 0.01, 0.04, -0.01]
# `sns.histplot()` is a function from the seaborn library in Python that is used to create a histogram
# plot. A histogram is a graphical representation of the distribution of a dataset. It displays the
# frequency of occurrences of different values in a dataset by dividing the data into bins and
# counting the number of values that fall into each bin.
sns.histplot(returns, bins = 5, kde=True)
plt.title("Distribution of Stock Returns")
plt.show()


Exercices on Basic Data
-----------------------

#1
total_shares = int(input("Numbers of shares you want to buy :"))
price = float(input("Share price :"))
total_value = total_shares * price
print(total_value)

#2
annual_rate_percentage = float(input("Annual Interest Rate in percentage :"))
annual_rate_float = annual_rate_percentage / 100
formatted_interest_rate = f"{annual_rate_float:.2f}"
print(formatted_interest_rate)

#3
stock_price = float(input("Stock prices :"))
final_price = round(stock_price)
print(final_price)

#4
compagnyname = input("What is the compagny's name ? ")
ticker = input("What is the compagny's ticker ?")
compagnyCEO = input("What is the compagny CEO name ?")
print("The compagny's name is ", compagnyname, ", its ticker is ", ticker, " and its CEO is ", compagnyCEO, "." )

#5
bond_rating = "AAA"
# == est un comparateur entre ce qu'il y a gauche et a droite du signe. 
#Si la valeur de bond_rating est égale à "AAA", la condition est vraie, et le code à l'intérieur du bloc if est exécuté.
if bond_rating == AAA :
    bond_rating = AA+
print(bond_rating)



Exercices on Collection 
-----------------------

#6
list_tickers = ["BTC", "ETH", "SOL"]
list_tickers.append("DODGE")
list_tickers.pop(1)
print(list_tickers)

#7
stock_info = {
    "ticker" : "ABC",
    "price" : 100,
    "volume" : 1000,
    "outstanding" : 1000000
}
print(stock_info)

#8
daily_closing_price = [103, 102, 110, 115, 112, 108, 112]
average_closing_price = sum(daily_closing_price)/ len(daily_closing_price)
average_closing_price = f"{average_closing_price:.4f}"
print(average_closing_price)


#9
dictionary = {
    "ticker": "ABC",
    "stock_info" : {
        "price" : 10,
        "market cap" : 1000000,
        "sector" : "technology",
    },
}
print(dictionary)


#10
tickers = ["ABC","DEF","GHI","JKL"]
daily = [1.2, -0.4, 1.3, 0.4]
dictionary = dict(zip(tickers, daily))
print(dictionary)


Exercices on Loops 
------------------

#11
stock_prices = [123, 126, 119, 118, 122, 127, 130]
for i in range(1, len(stock_prices)):
        price_change = stock_prices[0] - stock_prices[i]
        print(f"Day {i}: Price change = {price_change}")

#12
tickers = ["ABC","DEF","GHI","JKL"]
current_price = [123, 126, 119, 118]
dict = {}
for i in range (len(tickers)):
    dict[tickers[i]] = [current_price[i]]
print(dict)


#13
budget = 1000
stock_price = 100
total = 0
while budget >= stock_price:
    total += stock_price
    budget -= stock_price
nb_shares = total/ stock_price
print(nb_shares)

#14
rates = [0.32, 0.21, 0.05, 0.12, 0.08]
for i in rates:
    investment = 1000 * i
    print(f"With an interest rate of {i} the investment value is {investment}. ")

#15
import random
daily_closing_prices = []
daily_closing_prices = [random.uniform(50, 200) for _ in range(30)]
average_price = sum(daily_closing_prices) / len(daily_closing_prices)
print("Average Stock Price for the Month:", average_price)
while average_price >= 100:
    highest_price = max(daily_closing_prices)
    daily_closing_prices.remove(highest_price)
    print("Removed highest price: ",highest_price)
    average_price = sum(daily_closing_prices) / len(daily_closing_prices)
    print("New average is ", average_price)
print("Now the average is beyond $100")


EXERCICES ON CONDITIONAL STATEMENTS
-----------------------------------

#16
credit_loan = (646)
if credit_loan >= 700:
    print("Approve the loan")
elif credit_loan >= 650 :
    print ("Conditionally approve the loan")
else:
    print("decline the loan")


#17
dividend_yield = (0.07)
if dividend_yield > 0.05:
    print("high yield")
elif dividend_yield > 0.02:
    print("moderate yield")
else :
    print("low yield")

#18
stock_price = (120)
opening_price = (134)
if opening_price < stock_price * 0.95:
    print("buy")
elif opening_price > stock_price * 1.1:
    print("sell")


#19
bond_rating = str("AA+")
rating = ["AAA", "AA+", "BBB"]

if bond_rating in rating:
    print("investment grade")
else:
    print("non-investment grade")
    

#20
portfolio = (100)
loss = (0.03)

if loss >= 0.02:
    print("have to sell 10 percent of the total portfolio's asset")
    portfolio = portfolio * 0.9
    print("new portfolio's value is ", portfolio)
else: 
    print("no significant variation")



EXERCICES ON OBJECTED ORIENTED PROGRAMMING
------------------------------------------

#21
class Stock:
    def __init__ (self, ticker, base_price, current_price):
        self.ticker = ticker
        self.base_price = base_price
        self.current_price = current_price
    def method(self):
        change_price = (self.base_price - self.current_price) / self.current_price
        print(change_price)

#22
class Stock:

    def __init__ (self, ticker, base_price, current_price):
        self.ticker = ticker
        self.base_price = base_price
        self.current_price = current_price

    def calculate_value(self):
        return self.current_price

class Portfolio:

    def __init__ (self):
        self.stock = []

    def add_stock(self, stock):
        self.stock.append(stock)
        print("new stock add")

    def remove_stock(self, ticker):
        for stock in self.stock:
            if stock.ticker == ticker:
                self.stock.remove(stock)
		print("stock remove")
                break

    def total_value(self):
        total_value = 0
        for stock in self.stock:
            total_value += stock.calculate_value()
        return total_value
	print(total_value)

#23
class BankAccount:
    def __init__ (self, account_number, balance):
        self.account_number = account_number
        self.balance = balance
    
    def deposit(self, deposit_ammount):
        deposit_ammount = float(input("Deposit Ammount :"))
        if deposit_ammount > 0:
            self.balance += deposit_ammount
            print("New balance is :", self.balance)
        else :
            print("No deposit ammount.")

    def withdraw(self, withdraw_ammount):
        withdraw_ammount = float(input("Withdraw Ammount :"))
        if withdraw_ammount > 0 and withdraw_ammount <= self.balance:
            self.balance -= withdraw_ammount
            print("New balance is :", self.balance)
        elif withdraw_ammount > self.balance:
            print("Insufficient funds.")
        else :
            print("No withdraw ammount.")

    def check(self, balance):
        checkverif = input("Do you want to check the balance ? (Yes/ No):")
        if checkverif == "Yes":
            print(self.balance)
        elif checkverif == "No":
            print("no balance check")
        else : 
            print("Erreur Value")  



















