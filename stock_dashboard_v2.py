import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import datetime
import altair as alt
from streamlit_tags import st_tags

# Setting up the Streamlit app layout
st.set_page_config(layout="wide")
# st.title("Jim Burke's Portfolio App")
# st.write("")

class DataFetcher:
    """Handles data fetching and caching."""
    @staticmethod
    @st.cache_data
    def get_ticker_data(start_date, end_date, tickers):
        return yf.download(tickers, start=start_date, end=end_date)['Adj Close']

class Portfolio:
    """Manages portfolio data and analysis."""
    def __init__(self, tickers, start_date, end_date, benchmark='^GSPC'):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.data = DataFetcher.get_ticker_data(start_date, end_date, tickers)
        self.benchmark_data = DataFetcher.get_ticker_data(start_date, end_date, [benchmark])
        self.benchmark = benchmark
    
    # @st.cache_data
    def calculate_returns(_self):
        returns = _self.data.pct_change()
        cumulative_returns = (1 + returns).cumprod() - 1
        weights = [1/len(_self.tickers)] * len(_self.tickers)
        returns['Portfolio'] = returns.dot(weights)
        cumulative_returns['Portfolio'] = (1 + returns['Portfolio']).cumprod() - 1
        best_performers = cumulative_returns.iloc[-1].nlargest(3).index.tolist()
        worst_performers = cumulative_returns.iloc[-1].nsmallest(3).index.tolist()
        return cumulative_returns, best_performers, worst_performers
    
    @st.cache_data
    def calculate_sharpe_ratio(_self):
        returns = _self.data.pct_change().dropna()
        portfolio_returns = returns.mean(axis=1)
        benchmark_returns = _self.benchmark_data.pct_change().dropna()
        excess_returns = portfolio_returns - benchmark_returns
        sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252)  # Annualized Sharpe ratio
        return sharpe_ratio
    
    @st.cache_data
    def calculate_std_dev(_self):
        returns = _self.data.pct_change().dropna()
        portfolio_returns = returns.mean(axis=1)
        std_dev = portfolio_returns.std() * np.sqrt(252)  # Annualized standard deviation
        return std_dev

class Plotter:
    """Handles plotting and visualization."""
    @staticmethod
    def highlight_max(cell):
        if type(cell) != str and cell < 0:
            return 'color: red'

    @staticmethod
    def plot_best_worst(cumulative_returns, best_performers, worst_performers):
        fig, ax = plt.subplots()
        ax.grid(True)
        ax.scatter(best_performers, cumulative_returns.loc[:, best_performers].iloc[-1], color='green', label='Best Performers', marker='^', s=100)
        ax.scatter(worst_performers, cumulative_returns.loc[:, worst_performers].iloc[-1], color='red', label='Worst Performers', marker='v', s=100)
        ax.set_title('Best and Worst Performers')
        ax.set_xlabel('Stock')
        ax.set_ylabel('Cumulative Return')
        ax.legend()
        return fig

    @staticmethod
    def plot_summary(portfolio_combined, sp500, dates, labels):
        fig, ax = plt.subplots()
        portfolio_combined.plot(ax=ax, label='Combined Portfolio')
        sp500.plot(ax=ax, label='S&P 500')
        
        for date, label in zip(dates, labels):
            ax.axvline(date, color='red', linestyle='--')
            ax.text(date, ax.get_ylim()[1], label, rotation=90, verticalalignment='bottom')
        
        ax.set_title('YTD Performance: Combined Portfolio vs S&P 500')
        ax.set_xlabel('Date')
        ax.set_ylabel('Cumulative Return')
        ax.legend()
        return fig
    
class PortfolioApp:
    """Manages the Streamlit app."""
    def __init__(self):
        self.BOY = datetime.datetime(2024, 1, 1)
        self.sub_date_1 = datetime.datetime(2024, 2, 14)
        self.sub_date_2 = datetime.datetime(2024, 5, 15)
        
        # Sidebar date selector
        self.selected_date = st.sidebar.date_input("Select a date", datetime.date.today())
        
        self.last_friday = (self.selected_date - datetime.timedelta(days=self.selected_date.weekday()) + datetime.timedelta(days=4, weeks=-1))
        self.this_friday = (self.selected_date - datetime.timedelta(days=self.selected_date.weekday()) + datetime.timedelta(days=4, weeks=0))
        
        self.initial_portfolio = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META', 'V', 'NOW', 'NVDA', 'MELI', 'FRFHF', 'GOOG', 'SQ', 'MA', 'CNSWF', 'LRCX', 'ABNB', 'BRK-B', 'MKL', 'JMHLY', 'DHR']
        self.rev1_portfolio = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META', 'V', 'NOW', 'NVDA', 'MELI', 'FRFHF', 'GOOG', 'ROP', 'MA', 'CNSWF', 'LRCX', 'ABNB', 'BRK-B', 'MKL', 'JMHLY', 'DHR']
        self.rev2_portfolio = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META', 'V', 'NOW', 'NVDA', 'TOITF', 'FRFHF', 'GOOG', 'ROP', 'MA', 'CNSWF', 'LRCX', 'ODFL', 'BRK-B', 'MKL', 'JMHLY', 'DHR']
        
        self.portfolio_1 = Portfolio(self.initial_portfolio, self.BOY, self.sub_date_1)
        self.portfolio_2 = Portfolio(self.rev1_portfolio, self.sub_date_1, self.sub_date_2)
        self.portfolio_3 = Portfolio(self.rev2_portfolio, self.sub_date_2, self.selected_date)
        self.sp_and_nasdaq = Portfolio(['^GSPC', '^IXIC'], self.BOY, self.selected_date)

    def calculate_combined_performance(self):
        combined_data = pd.concat([
            self.portfolio_1.data,
            self.portfolio_2.data,
            self.portfolio_3.data
        ], axis=1)
        
        max_value_and_date = combined_data[combined_data.columns.tolist()].idxmax()#:combined_data[combined_data.columns.tolist()].max()}

        combined_returns = combined_data.pct_change().mean(axis=1)
        cumulative_combined_returns = (1 + combined_returns).cumprod() - 1
        return cumulative_combined_returns, max_value_and_date
    
    def make_tabs(self, portfolio, start_date, end_date):
        st.dataframe(portfolio.data)
        cumulative_returns, best_performers, worst_performers = portfolio.calculate_returns()
        fig = Plotter.plot_best_worst(cumulative_returns, best_performers, worst_performers)

        # Fetch benchmark data
        benchmark_data = Portfolio(['^GSPC'], start_date, end_date).data
        sp_and_nasdaq_data = benchmark_data.pct_change().cumsum()


        # Prepare data for area and line charts
        cumulative_returns_df = cumulative_returns[['Portfolio']]
        combined_df = pd.DataFrame({
            'Portfolio': cumulative_returns_df['Portfolio'],
            'S&P 500': sp_and_nasdaq_data
        }).dropna()

        
        # Create Altair chart
        base = alt.Chart(combined_df.reset_index()).encode(
            x='Date:T'
        )

        area = base.mark_area(opacity=0.3, color='lightblue').encode(
            y='S&P 500:Q'
        )

        line = base.mark_line(color='blue').encode(
            y='Portfolio:Q'
        )

        chart = (area + line).interactive()

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("Portfolio Performance Versus S&P 500")
            st.altair_chart(chart, use_container_width=True)
        with col2:
            st.pyplot(fig, figsize=(4, 4))

        st.markdown("### Cumulative Daily Returns: Holding Period Return")
        st.dataframe(cumulative_returns.style.applymap(Plotter.highlight_max))


        cumulative_returns['S&P 500'] = sp_and_nasdaq_data
        cumulative_melted = cumulative_returns.reset_index().melt('Date', var_name = 'Ticker', value_name = 'Cumulative Return')
        # st.dataframe(cumulative_melted)
        # Create Altair chart
        base = alt.Chart(cumulative_melted, height = 600).encode(
            x='Date:T'
        )

        area = base.mark_area(opacity=0.4, color='lightblue').encode(
            y='Cumulative Return:Q'
        ).transform_filter(
            alt.FieldEqualPredicate(field='Ticker', equal='S&P 500')
        )

        line = base.mark_line().encode(
            y='Cumulative Return:Q',
            color='Ticker:N'
        ).transform_filter(
            {'not': alt.FieldEqualPredicate(field='Ticker', equal='S&P 500')}
        ) 

        chart = (area + line).interactive()

        st.markdown("### By-Stock Cumulative Returns Versus S&P 500")
        st.altair_chart(chart, use_container_width=True)

    def run(self):
        st.title("Jim Burke's Portfolio App")
        st.markdown("### Current Week Performance")
        st.write(f"Sharpe Ratio (Current Portfolio): {self.portfolio_3.calculate_sharpe_ratio():.2f}")
        st.write(f"Standard Deviation (Current Portfolio): {self.portfolio_3.calculate_std_dev():.2f}")


        col1, col2 = st.columns(2)
        with col1:
            cumulative_returns, best_performers, worst_performers = self.portfolio_3.calculate_returns()
            weekly_returns = (self.portfolio_3.data.iloc[-1] - self.portfolio_3.data.loc[str(self.last_friday)]) / self.portfolio_3.data.loc[str(self.last_friday)]
            performance_table = pd.concat([weekly_returns * 100, cumulative_returns.iloc[-1] * 100], axis=1, keys=['Weekly Return %', 'Cumulative Return %'])
            st.dataframe(performance_table.style.applymap(Plotter.highlight_max), width=400)
        with col2:
            current_week_data = self.sp_and_nasdaq.data.loc[str(self.last_friday):str(self.selected_date)]
            returns = current_week_data.pct_change().cumsum()
            # returns = self.current_week_data.data.pct_change()
            cumulative_returns = (1 + returns).cumprod() - 1
            weights = [1/2] * 2
            returns['Portfolio'] = returns.dot(weights)
            cumulative_returns = pd.DataFrame(cumulative_returns)
            weekly_returns = (self.sp_and_nasdaq.data.iloc[-1] - self.sp_and_nasdaq.data.loc[str(self.last_friday)]) / self.sp_and_nasdaq.data.loc[str(self.last_friday)]
            performance_table = pd.concat([weekly_returns * 100], axis=1, keys=['Weekly Return %'])
            st.dataframe(performance_table.style.applymap(Plotter.highlight_max), width=400)

            # Plot current week's performance of S&P500 and NASDAQ
            st.line_chart(current_week_data)

        st.header("Year-to-Date Performance Metrics")
        
        # Prepare data for area and line charts
        combined_returns, max_values = self.calculate_combined_performance() * 100
        sp500_returns = self.sp_and_nasdaq.data[self.sp_and_nasdaq.benchmark].pct_change().cumsum() * 100
        # Date slider
        date_range = st.slider("Select Date Range", min_value=self.BOY.date(), max_value=self.selected_date, value=(self.BOY.date(), self.selected_date))
        filtered_combined_returns = combined_returns.loc[date_range[0]:date_range[1]]
        filtered_sp500_returns = sp500_returns.loc[date_range[0]:date_range[1]]

        combined_df = pd.DataFrame({
            'Portfolio': filtered_combined_returns,
            'S&P 500': filtered_sp500_returns
        }).dropna()

        
        # Create Altair chart
        base = alt.Chart(combined_df.reset_index()).encode(
            x='Date:T'
        )

        line1 = base.mark_area(opacity=0.3, color='lightblue').encode(
            y='S&P 500:Q'
        )

        line2 = base.mark_line(color='blue').encode(
            y='Portfolio:Q'
        )

        rules = alt.Chart(pd.DataFrame({
        'Date': ['2024-02-14', '2024-05-15'],
        'color': ['red', 'orange'],
        'text': ['1st Iteration of Portfolio', '2nd Iteration of Portfolio']
        })).mark_rule().encode(
        x='Date:T',
        color=alt.Color('color:N', scale=None),
        text = 'text'
        )

        text = rules.mark_text(
            align="left",
            baseline="middle",
            dx=3
        ).encode(x='Date', text = 'text')
  
        chart = (line1 + line2 + rules + text).interactive()
        st.altair_chart(chart, use_container_width=True)

        tab1, tab2, tab3 = st.tabs(["Current Portfolio", "Original Portfolio", "Second Portfolio"])
        with tab1:
            st.header("Current Portfolio")
            st.write("This is the current iteration of the portfolio which began on 5/16/2024. This 3rd iteration consisted of swapping out **ABNB** and **MELI** for **TOITF** and **ODFL**.")
            self.make_tabs(self.portfolio_3, self.sub_date_2, self.selected_date)
        with tab2:
            st.header("Original Portfolio")
            st.write("This is the first iteration of the portfolio which existed from the beginning of the year until 2/14/2024.")
            self.make_tabs(self.portfolio_1, self.BOY, self.sub_date_1)
        with tab3:
            st.header("Second iteration of the portfolio with stock substitutions.")
            st.write("This is the second iteration of the portfolio which existed from 2/15/2024 to 5/15/2024 and saw the substitution of **SQ** with **ROP**.")
            self.make_tabs(self.portfolio_2, self.sub_date_1, self.sub_date_2)


# Run the app
if __name__ == "__main__":
    app = PortfolioApp()
    app.run()
