# Visualizing the stock market structure with Scikit-learn

This project follows closely the [example](https://scikit-learn.org/stable/auto_examples/applications/plot_stock_market.html#sphx-glr-auto-examples-applications-plot-stock-market-py) provided by `Scikit-learn`!

The project employs unsupervised learning techniques to extract the stock market structure from variations in historical quotes.

## Retrieve data from Internet

The data is from 2003 - 2008, and represent the daily variation in quote prices.

## Code structure

We use `Pandas` to store the data, and `Numpy` to process them. And `Matplotlib` to create graphics!

In module `stock_market_structure.py` we load the data.
In module `stock-plotter.py` we visualize the quote variation after clustering quotes according to their covariance.

`Scikit-learn` helps us a lot!
**Fascinating** example, _don't you think_!

Just run:
~~~
python stock-plotter.py
~~~

Take a look at the expected outcome, which is `stock_market_structure.png`!
