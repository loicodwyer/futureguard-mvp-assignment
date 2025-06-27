# FutureGuard – Personal Financial Risk Analysis MVP

## Project Overview

**FutureGuard** is a new FinTech MVP that provides personal financial risk analysis within the framework of a banking application. It allows users to forecast their cash flow and understand potential risks in finances. Unlike the static linear projections of conventional personal finance tools, FutureGuard uses **probabilistic forecasting based on Machine Learning and Monte Carlo simulations**. This means that it simulates many possible scenarios regarding a person's future financial situation and calculates the downside risk by using **Value at Risk (VaR)**. Its aim is to help users prepare for uncertainty by showing a range of outcomes (best case, worst case, most likely scenarios) and to offer personalized insights and recommendations based on these outcomes.

This strategy fills a current gap in European retail banking applications and FinTech, since there is no European retail banking application or FinTech solution available that combines Monte Carlo risk modeling and Value at Risk (VaR) with Machine Learning models for personal financial management by consumers. By closing this gap, FutureGuard allows consumers to use institutional-level risk management tools (traditionally used by banks and investors) as part of their daily finances. The key features of the platform are:
- **Probabilistic Risk Dashboard**: The user receives a set of potential future balances (highs, lows, and medians) rather than a point forecast, enabling them to plan for uncertainty.
- **Monte Carlo Simulation Engine**: The model simulates random variations in income and expenses to create numerous potential future cash flow scenarios.
- **Value at Risk (VaR) Measure**: It calculates the loss in financial terms at a confidence level (e.g. "5% chance your balance falls below €X") to quantify downside risk.
- **Machine Learning Predictions**: A Transformer-based ML model analyzes historical transactions to forecast future cash flow trends, increasing the accuracy and personalization of the simulations.
- **Personalized Insights & Recommendations**: The application provides AI-driven insights (e.g. likelihood of shortfall) and actionable recommendations (e.g. spend less, saving tips) tailored to the user's situation. 
- **Integrated User Interface within Banking Application**: Built for integration into a mobile banking application (e.g., ING Bank), thereby enabling users to obtain risk assessments alongside their usual account details without the need for multiple logins or data exports.

Overall, FutureGuard seeks to provide probabilistic, forward-looking financial planning to everyday users in order to enable them to make sound decisions and proactively deal with risks in their personal finances.

## Architecture Overview

System Architecture: FutureGuard is a client-server application with a Python **FastAPI** backend and a **Flutter** frontend. All data processing, machine-learning predictions, and simulation logic are handled in the backend, while a user-friendly interface and visualisations are provided by the frontend. The two communicate using RESTful API calls over HTTP.

In the **backend**, the components are:
- **FastAPI Server:** Allows exposing REST endpoints to execute risk analyses and retrieve data. It manages the end-to-end process upon receiving a request (e.g., `/run_risk_analysis/{user_id}`).
- **Data Layer:** For this MVP, user-transaction data is stored in CSV files (synthetic dataset named `synthetic_transactions.csv`). In a live deployment this would be a database, but here the backend simply loads up the CSV data.
- **Machine-Learning Module:** A Transformer-based model (built using TensorFlow/Keras) is used in the backend for predicting future cash flows. The model uses the user's historical daily net cash flow (income minus expenses) found and various engineered features to predict future financial trends, in a built dataset called `synthetic_transactions_daily_enriched`. With a Transformer (using multi-head attention), the model can learn temporal patterns and seasonality from the user's transactions more efficiently than a static model. 
- **Monte Carlo Simulation:** The backend takes the machine learning model's predictions as a base expectation, then runs a Monte Carlo simulation to introduce elements of randomness and volatility. Practically speaking, this will involve the backend generating simulations of 10,000 potential future trajectories of the user's financial situation over the time horizon. This is done by introducing stochastic variability into the projected values, drawn from distributions derived from the Transformer model's predictions. This captures real-world uncertainty such as variable incomes, unexpected expenses, or random fluctuations. The simulation provides a distribution of results for each future time period.
- **Value at Risk Calculation:** Following simulations, the backend goes ahead to calculate **VaR** statistics. For instance, the 5th percentile of the ending balance distribution may serve as the 95% Value at Risk (i.e., a 5% chance that the user's balance will fall below this point at the end of the specified horizon). This risk metric has the capacity to provide informative numbers from the extensive simulation results. 
- **Insights and Recommendations Engine:** Drawing upon the results of the simulation and associated risk metrics, the backend produces several insights in a comprehensible format along with recommended strategies. These are currently rules-based (for the MVP) – i.e., if there is a strong possibility of overdraft, the system can generate an insight like " In 5 % of cases, your balance will go below 0." and a recommendation like "Deposit an extra €X to avoid negative balance in worst-case scenarios." The insights and recommendations are then returned as part of the API response.

In the **frontend (Flutter app)**, the components are:
- **State Management:** The app uses the **Provider** package for state management. There are providers for things like the selected user and date range (**OverviewState**) as well as for the risk-analysis result (**ScenarioState**).  These supply the UI state and trigger API requests. For example, the OverviewState holds the current `userId`, `start date`, and `horizon` (forecasting time span in days) that are selected.
- **User Interface (UI) Pages:** The application is organized by a tabbed navigation system (**BottomNavigationBar**). The primary pages are **Overview** and **Risk Dashboard:**
    - **Overview Page:** Shows the user's current account overview and recent transactions.  It includes controls to select the user ID (where we can switch between several distinct user profiles), select a start date, and select a forecast range (e.g. 30, 90, 180 days) to examine.  These options are only included for presentation purposes as users of course would not be able to view other users' account. As the user changes these, the app fetches the updated overview from the backend. The dashboard displays the user's identification, International Bank Account Number (IBAN), current financial balance as of the chosen start date, and a summary list of recent transactions up until the selected date. It also includes a group of four "widgets" or quick action chips. These closely resemble current bank account views, and provide common bank account actions. In the Minimum Viable Product (MVP), these are simply text chips for illustrative purposes. 
    - **Risk Analysis Page:** The other page, titled the **Risk Dashboard**, serves as the entry point for the user to execute and display the risk analysis. At first sight, it contains a *"Run analysis"* button. When clicked, the application calls the `/run_risk_analysis` API and subsequently displays a loading symbol. Upon receiving the results, the page presents a **scrollable dashboard** with:    
        - A **Fan Chart** graph of 3 months' history and forecast distribution (blue for history, orange for median future, shaded area for P5–P95 range, dashed for best/worst). The historical data, that is in the past 90 days prior to the start date, is represented as a line for context. In contrast, the forecast horizon is represented by a median result line with a shaded confidence band ranging from the 5th to the 95th percentile of the runs. The app employs the `fl_chart` package to plot this line chart that accommodates multiple data series. For instance, the median forecast can be represented by an orange line, with the region between the 5th percentile (p5) and the 95th percentile (p95) outcomes indicated by a semi-transparent fill to represent uncertainty. Furthermore, the **optimal** and **suboptimal** cases of the simulations can be represented using dashed lines to reveal the most extreme trajectories. A legend is included to describe the various lines in the graph (e.g., median compared to range). Users can click on the chart to view tooltips with precise values for a specified date (leverage fl_chart's features for interactive tooltips). 
        - An **Insights panel**: a card with a few key insights, where each insight is a one-line sentence. For example, an insight can say "There is a 95% likelihood your balance falls below €0 in the next 3 months". These give the user context to what the numbers are.    
        - A **Recommendations panel**: a bullet-point list of recommendations or action items produced by the system. These are individualized suggestions designed to enhance the user's financial situation or reduce potential risks. Examples of proposals are along the lines of phrases like "Raise your account balance by €5,000 to build a substantial safety buffer (this would eliminate overdrafts in 95% of situations)". Three proposals are presented in this panel. Currently, a rule based method is for both the insight and recommendations. However, an LLM (such as GPT-4) should be created and used to provide the best possible messages. 
        - A **Download Report button**: This functionality allows the user to download a full report related to the risk analysis. Triggering this functionality should call the `/download_risk_analysis` API, which in turn creates and returns a ZIP file. The app is then supposed to save this file on the device by using Flutter's file APIs. The ZIP contains more details on the outputs of the analysis, i.e., the raw simulation data (CSV of all simulated runs) and a snapshot graph of the forecast. Regrettably, because of time limitations and a shortage of technical expertise on my part, the download button in the present scenario doesn't create CSV files and snapshots. In a real-life deployment, this feature would enable reports to be directly downloaded to device used, with email or share features to also be developed. 


The overall data flow is as follows: 
1. the user interacts with the Flutter app (selecting options and requesting analysis);
2. the app makes requests to the FastAPI backend
3. the backend loads data and model, computes forecasts and simulations
4. results (JSON data and insights) are returned to the app 
5. the app renders the data in charts and text for the user. 

This separation allows for heavy computations to be done on the server side, which can be scaled or optimized according to requirements, and the user interface is responsive and dedicated to enhancing user experience.

Here, in this MVP, both frontend and backend are run locally (during development). The backend is run on a local server (default `127.0.0.1:8000`), and the Flutter app is configured to send API requests to this URL. This can be set using a base URL configuration in the app in case of running on a remote server. The architecture is designed so that the backend could eventually be containerized and deployed on the cloud, and the Flutter app could be shipped to consumers via app stores, communicating with the cloud API. There are no external third-party services required for core functionality. All ML and simulation logic is self-contained.

## Technological Framework

FutureGuard uses contemporary Python and Flutter/Dart technologies. The key elements of the technological stack are:

- **Python 3.10**: A programming language for backend development.
- **FastAPI**: Web framework to build the REST API backend. Chosen due to its performance and easy Pydantic integration for data models.
- **Uvicorn**: ASGI server for testing and local development of the FastAPI application.
- **Data Science Libraries** such  as **Pandas** and **NumPy** to manipulate data (load CSVs, compute rolling statistics, etc.), and **Matplotlib** for plotting (to generate any charts for downloadable report).
- **Machine Learning**: **TensorFlow / Keras** for implementing the Transformer model. The model uses Keras layers, like `MultiHeadAttention`, to build an encoder-decoder or sequence prediction network for time-series prediction.
- **SciPy/Stats or scikit-learn**: Applied in any statistical calculations or data preprocessing (e.g., for creating random variables for simulation or calculating percentiles, although NumPy can also calculate percentiles).
- **Python ZIP and I/O libraries**: used to compress the CSV and image within a ZIP file for response at the `/download_risk_analysis` endpoint.
- **Flutter (Dart)**: The cross-platform frontend application framework. This allows for a shared codebase between Android and iOS (and even web/desktop if required). Flutter was chosen because of its fast development cycle and expressive UI support.
- **Provider**: A Flutter state management library used to manage app state (selected user, fetched data, analysis outcome) in an easy, reactive way.
- **HTTP (Dart package)**: Used by the Flutter app to make REST requests to the FastAPI backend and obtain JSON responses.
- **fl_chart**: A Flutter charting library through which line charts are drawn to show historic balance and simulation results. It provides multiple series support and customization of the styling, which is used to construct the fan chart (median line, percentile bands, etc.) as well as legends.
- **Intl (Dart)**: Internationalization and number formatting (used for formatting in the UI).
- **Path & PathProvider (Dart)**: Utility classes for file paths and saving files (e.g. to save the downloaded ZIP report on the device in a readable location).
- **Dart Dev Tools**: Dev dependencies of Flutter are `flutter_test` (for writing unit tests if needed). 

All of these libraries and tools work together to provide a powerful yet simple-to-implement MVP. FastAPI and Flutter ensure the system is reactive and lightweight, and numpy/pandas and TensorFlow allow for leveraging ML and analytical functionality. 

## Setup and Installation

To run the FutureGuard project on a local machine, you need to set up both the backend (FastAPI) and the frontend (Flutter app). Below are step-by-step instructions for each:

### Backend Setup (FastAPI)

1. **Prerequisites:** Make sure you have **Python 3.9+** installed on your system. It is also recommended to use a Python virtual environment to avoid dependency conflicts.  
2. **Project Files:** The backend source code is contained in the `futureguard_api` directory along with data files (`synthetic_transactions_daily_enriched.csv`, `synthetic_transactions_enriched.csv`, and `synthetic_transactions.csv`) and notebooks for data/model prep (found in the folder within `futureguard_api` titled `data generation`). The main python files used in the backend are called `main.py` and `risk pipeline.py`. 
3. **Install Dependencies:** Inside a terminal, navigate to the backend project directory. Install the required Python libraries:
 ```bash
pip install fastapi uvicorn pandas numpy tensorflow matplotlib
```
4. **Prepare Data**: Ensure that the synthetic data CSV files (generated in the Data Generation section) are present in the expected location, which is within the `futureguard_api` folder. 
5. **Run the API Server**: Launch the FastAPI server using Uvicorn. This means you are required to open the folder in VS Code and paste `uvicorn main:app --reload` in the terminal to start the API. The server should start on http://127.0.0.1:8000 by default. Again, below is the code to run in the terminal.
```bash
uvicorn main:app --reload
```
6. **API Docs**: FastAPI automatically provides interactive API docs. You can visit http://127.0.0.1:8000/docs to view and test the endpoints in a web UI (Swagger UI), which is helpful for verification.


## Frontend Setup (Flutter)

1. **Prerequisites**: Install Flutter SDK (version 3.8 and above). Refer to the official Flutter installation guide for your platform. Ensure you have been able to execute `flutter doctor` with no major issues detected.
2. **Project Files**: Flutter application source code is in the `futureguard_app` directory. You can open this folder using an IDE like Android Studio, VS Code, or run terminal commands.
3. **Flutter Packages**: Navigate to the `futureguard_app` folder and execute `fetch dependencies`. This will get all the pub packages (provider, http, fl_chart, etc.) that are listed in pubspec.yaml. Here is the code to write in the terminal: 
```bash
flutter pub get
```
4. **Configure Backend Endpoint**: The app is preconfigured to use http://127.0.0.1:8000 for its API calls (see `ApiClient.baseUrl`). This setup works when running the app in a web browser or an emulator that is on the same host as the backend.
    - When you're running the Flutter app on an Android emulator, remember that `127.0.0.1` on the emulator points to the emulator, not the host machine. You must change the baseUrl to http://10.0.2.2:8000 (a special alias to host machine from Android emulator). You can do this by modifying the code in ApiClient or modifying configurations as appropriate.
    - In case you run the Flutter application in Chrome (web debug mode) or an iOS simulator on the same computer, 127.0.0.1:8000 will be fine. For a physical device, you would have to point to your computer's IP address if both are on the same network, or host the backend to a server that the device can access.
5. **Run the App**: Run via Flutter CLI or IDE. Choose the target device (e.g., web with Chrome, an open emulator, or a connected device), which we recommend to do in with chrome. This will compile the app and show it. You should now be able to see the FutureGuard app with its Overview page on the device/emulator.
```bash
flutter run

or 

flutter run -d chrome
```

6. **Running the App**: In the app, select a user ID, date, and horizon, then navigate to the Risk tab and click on "Run analysis". Ensure your backend is running to serve the request. You should see the analysis results populate the UI. 

*Note:* The Flutter application is now set up in debug mode. There are no release build configurations available (since deployment is not configured). To create a release build (APK or IPA), you might need to add necessary app icons, permissions (if any), etc., but for MVP demonstration this is not required.

## Data Generation

For this FutureGuard MVP project, **synthetic transaction data** is used to simulate user financial histories. The project includes scripts (Jupyter notebooks) that generate (using the Faker package) and enrich the data. If you want to understand or reproduce the data generation process, please refer to the notebooks in the repository (in the `data generation` folder within `futureguard_api`). The overall steps are to generate raw transactions for numerous users and then derive additional features on this data to use in the machine learning model.

To generate the dataset from scratch, you can run the notebook dataset_generation_code.ipynb (and subsequent notebooks for adding features). The notebooks will give you CSV files with the following contents:
- `synthetic_transactions.csv`: This represents the **Transaction-level data** dataset generated in the python script titled `dataset generation code.ipynb` using the Faker package. This dataset includes raw transaction data for 100 fictional users over a period (Jan 2022 through mid-2025). Each row represents a single transaction. Key columns include:
    - `date`: the transaction date (e.g., 2022-01-01).
    - `user_id`: an identifier for the user (1 through 100).
    - `amount`: the transaction amount (positive for income, negative for expense).
    - `description`: a description or merchant name.
    - `category`: a high-level category for the transaction (Income, Rent, Groceries, Transport, Utilities, etc.).
    - `name`: the name of the user (a fictional name).
    - `age` and `age_bracket`: the user's age and an age range bucket.
    - `country`, `city`: the user’s location (for realism and simplicity, all are within the Netherlands in this dataset).
    - `iban`: a fake IBAN number for the user's account.
    - **Temporal flags**: `season` (e.g., Winter) derived from date, `is_weekend` (True/False), `is_public_holiday` (True/False) indicating if the date was a weekend or holiday. These add context to spending patterns.

This raw dataset provides the base transaction history for each user.

- `synthetic_transactions_enriched.csv`: This represents the **Transaction-level data with additional features** dataset generated in the python script titled `dataset added variables.ipynb`. This builds on the raw transactions with a number of rolling statistics and indicators that help with modeling. The new columns introduced are:
    - **Rolling Window Stats** which include `rolling_7d_sum`, `rolling_7d_mean`, `rolling_7d_std` (and similar for 30 days and 90 days): these are rolling sums, averages, and standard deviations of spending over the past 7, 30, or 90 days up to that transaction’s date. They are included as they can help capture recent spending trends and volatility.
    - `days_since_last_tx`: days since the last transaction.
    - `cum_month_spend`: cumulative spending in the current month up to that date (and resets each month).
    - `is_payday`: flag indicating if this transaction date is likely a payday (e.g., salary credit). 
    - `time_since_payday`: number of days since the last payday.
    - **Spending Category Ratios** which include `ratio_Groceries`, `ratio_Rent`, `ratio_Entertainment`, etc.: These represent the proportion of the user’s spending ( in a month) that each category accounts for. These ratios help to personalize the model to each user's spending habits (i.e., a user who spends 40% on rent compared to one who spends 0% on rent if they don't have rent as an expense).
    - `day_of_month`: the day number (1–31) of the transaction date, included to help detect monthly patterns (like higher spending at start or end of month).

Each transaction row now has rich contextual features. A large part of these features are used as inputs to the machine learning model in order to enhance its predictions. Indeed, for instance, rolling mean and standard deviation could potentially assist the model in assessing volatility, while category ratios can assist in deciphering the composition of spending.
- `synthetic_transactions_daily_enriched.csv`: This represents **daily aggregated data with added features** dataset generated in the python script titled `dataset added variables v2.ipynb`. This code aggregates the transactions on a daily basis for each user, which is particularly useful for time-series models. One row represents a specific day per individual user and contains most of the same variables as `synthetic_transactions_enriched.csv` just on a daily basis. This CSV file therefore includes :
    - `user_id`, `date`: which user and which date the row represents.
    - `income` and `expenses`: the total income and total expense for that user on that date (summing all transactions of the day by sign).
    - `net_cash_flow`: simply `income + expenses` (since expenses are negative, this is effectively the net change in balance for that day).
    - All the same features as the previously mentioned file (rolling sums, is_weekend, etc.) but computed on a daily basis. For example, `rolling_7d_sum` here would be the sum of net cash flow for the last 7 days.

The daily data is what is fed into the ML forecasting model. With daily net cash flow as the time series, the model is able to forecast the net flow for future days, which can then be combined with the most recent known balance in order to make estimates of future balances.

The time series data is the foundation for all that follows in the `risk_pipeline.ipynb` file. The ML model is trained on the daily enriched data to learn patterns of how net cash flow behaves given various features (day of month, recent trends, etc.). This allows it to predict future net cash flows for a given user. When the backend is running a risk analysis for a user, it takes the historical data for that user (from `synthetic_transactions_daily_enriched.csv`) up to but not including the requested `start` date. Then it applies the Transformer model to forecast the net cash flow over the next `horizon` days. These predictions direct the Monte Carlo simulation.



## Testing and Model Scripts

The project also provides Jupyter notebooks that demonstrate the testing of code for the machine learning and simulation codes. These can be used for further development, experimentation, or verifying the algorithms outside the app’s UI. In the end, those test scripts are nearly exactly the same as the one found in `risk_pipeline.ipynb`, but presented as separate jupyter notebooks that do not include links to `futureguard_app`. The key scripts/notebooks include:
- `transformer code v4 final.ipynb`: This notebook contains code to train the Transformer-based time series model and make predictions based on the historical data. This Transformer-Based Cash-Flow Forecasting notebook covers loading the data (likely the daily enriched dataset), defining the model architecture, compiling and training the model, and evaluating its performance. Here are some more details about the steps of this notebook: 
    - Data Ingestion: 
        - Input file: `synthetic_transactions_daily_enriched.csv`  
        - Sampling: the notebook selects one `USER_ID` (default = 10).  
        - Cleaning: missing days are forward-filled with zeros so the time series is strictly daily.  
        - Scaling: amounts are divided by the inter-quartile range to improve numerical stability.
    - Model Architecture includes encoder-only Transformer with: 
        - 4 self-attention heads  
        - Model dimension \(d_{\text{model}} = 64\)  
        - Position encodings added with sine–cosine scheme  
        - Two-layer feed-forward block (\(128 \rightarrow 64\)) and layer normalisation  
        - The outputs are:  
            1. probability of a salary event (binary cross-entropy)  
            2. three quantile regressors (p10, p50, p90) for net amount (pinball loss)  
    - Training Protocol  
        - Split: last 20 % of days held out as a temporal test set.  
        - Optimiser: Adam, learning rate \(1\times10^{-3}\).  
        - Epochs: 50 with early-stopping (patience = 7).  
        - Seeds: NumPy and TensorFlow both fixed to 42 for reproducibility.  
    - Final output:
        - Forecast CSV outputed as `final_income_expense_predictions_v2.csv`, which as the following columns: `date`, `user_id`, `income_prob`, `income_p10`, `income_p50`, `income_p90`, `expense_p10`, …  


- **Monte Carlo & VaR Simulation Notebook** (e.g. `monte carlo and VaR code.ipynb`): This notebook demonstrates the risk simulation process. It uses either the trained ML model or some statistical approximation to project future scenarios. Main steps in this notebook include:
    - Loading a user’s historical data and computing their current balance.
    - Using the ML model predictions to get an expected future cash flow series.
    - Running many simulation runs (e.g., 10,000) where random noise is added to the expected values to simulate different scenarios. 
    - Storing the simulation results. 
    - Plotting charts for visualization.
    - Computing Value at Risk. Indeed, the notebook explicitly calculates VaR. It shows how to calculate this from the simulation data (typically by taking the appropriate percentile of losses).
    - Here is a general idea of the code structure: 
        - The initial Conditions are: 
            - Starting balance is reconstructed from the same transactions and frozen at **2025-02-28**. This can however be adjusted to the preference of starting of the tester. 
            - Forecast file path is set in the CONFIGURATION block (defaults to the CSV produced by the previously mentioned Notebook of ML predictions).
        - The stochastic engine includes:   
            - Path Generation: 
                - Horizon: 92 calendar days (2025-03-01 → 2025-05-31).  
                - For each day the algorithm draws: 
                    - an income indicator from Bernoulli,  
                    - income and expense magnitudes from triangular distributions bounded by the p10 and p90 quantiles.  
            - Repetition: 
                - Number of trials: 10 000 (configurable).  
                - Each path is cumulatively summed to yield a daily balance trajectory.
    - The main outputs from this code are: 
        - `full_simulation_output.csv` — 10 000 × 92 matrix of balances.  
        - Two PNG figures saved locally:  
            - Fan chart of all simulated balances (median line plus 5–95 % band). 
            - Return histogram after 92 days. 
        - Natural-Language Interpretation: Simple rule-based logic which converts the metrics into human-like advice. 

By running this notebook, you can observe the intermediate results of the pipeline. It effectively serves as a testing code for the core risk analysis logic.

**Running the notebooks**: To use these notebooks, you should make sure to have Jupyter or an environment like VSCode that can run .ipynb files. Make sure to install the necessary Python packages (especially for the ML notebook, you’ll need TensorFlow). Then open the notebook and run through the cells. The notebooks are documented with markdown and outputs to guide you. They are primarily for development/testing purposes and are not needed to run the app. The codes will output a CSV containing all simulated scenario data for further analysis, and you can always save plot images. These outputs mirror what the app’s features provide (i.e. chart of the simulation and VaR). By examining them, a developer or tester can verify correctness (for example, ensure that the 5th and 95th percentile in the CSV match what the app displays in the chart). The files `risk_pipeline.ipynb` and `main.ipynb` are both the most important files for the backend (`futureguard_api`)


## API Usage (Backend Endpoints)

The FastAPI backend provides several REST endpoints that the frontend (or any client) can call. The points listed below can be found in `main.py` within the `futureguard_api` folder. Below is a list of the available endpoints, including their purpose, parameters, and expected responses:
- ` GET /ping`: This is a health check endpoint. This is a simple endpoint to verify that the server is running. It takes no parameters. Response: JSON object {"ok": true} if the API is up. (The Flutter app calls this on startup to ensure connectivity.)
- `GET /account_overview/{user_id}`: Here we fetch account overview data for a slected user.
    - **Path Parameters**: `user_id` (integer ID of the user whose data to fetch).
    - **Query Parameters**:
        - start (date in YYYY-MM-DD format) = The starting date for the overview. Transactions up to this date will be considered. This will be the start date for prediction. 
        - horizon (integer, number of days) = The forecast horizon. How many days into the future do we want to predict. 
    
    Response: JSON object containing the user's overview info. This includes a `user` section (ID, name, iban), the current account `balance` as of the start date (after all transactions up to that date), a list of `widgets` (strings for suggested actions on the overview), and a list of recent transactions (`tx`). Each transaction has at least a date, a description, and an amount (where positive is credit, negative is debit). The Flutter app uses this data in the Overview page screen. For example, the code could output something like what is shown below:
    
    ``` json
    {
    "user": {
        "id": 10,
        "name": "A. SCHELLEKENS",
        "iban": "NL53HEXD8196001338"
    },
    "balance": 1234.56,
    "widgets": ["Invest Surplus", "Insurance Check-Up", "Budget Advice", "Savings Plan"],
    "tx": [
        {
        "date": "2025-02-25",
        "counterparty": "Albert Heijn",
        "amount": -45.20
        },
        {
        "date": "2025-02-24",
        "counterparty": "Salary Payment",
        "amount": 3000.00
        },
        ...
    ]
    }
    ```
- `POST /run_risk_analysis/{user_id}` = Run the risk analysis pipeline for a user.
This is the main endpoint that performs the ML prediction, Monte Carlo simulation and Value at Risk analysis. It may take a few seconds to compute, depending on the complexity (in the MVP it should be fairly quick).
    - Path Parameters: user_id (integer ID of the user).
    - Query Parameters:
        - `start` (date `YYYY-MM-DD`) = The starting date for the analysis. 
        - `horizon` (integer) = Number of days to forecast into the future. E.g., 30 for one month, 92 for a quarter. The model will project this many days of net cash flows, and simulations will cover this range.

    A call to this endpoint might look like:

    ```nginx
    POST http://127.0.0.1:8000/run_risk_analysis/10?start=2025-03-01&horizon=92
    ```
    This would trigger a 92-day (approximately 3 months) analysis for user 10 starting from March 1, 2025. The response JSON will contain all data needed for the Flutter app to display the risk dashboard for Q2 2025 for user 10.

    Response: JSON object with the simulation results and generated insights. The structure is as follows:
    ```json
    {
    "hist_dates": [...],       // e.g. ["2025-01-01", ..., "2025-03-01"]
    "hist_balances": [...],    // historical daily balance corresponding to hist_dates
    "dates": [...],            // future dates for each day of the horizon (e.g. ["2025-03-02", ..., "2025-06-01"])
    "median": [...],           // median projected balance for each future date
    "p5": [...],               // 5th percentile (worst-case band) balance for each date
    "p95": [...],              // 95th percentile (best-case band) balance for each date
    "best": [...],             // one example of the best-case scenario trajectory (could be the maximum run)
    "worst": [...],            // one example of the worst-case scenario trajectory (could be the minimum run)
    "insights": [
        "Insight sentence 1...",
        "Insight sentence 2...",
        "Insight sentence 3..."
    ],
    "recommendations": [
        "Recommendation 1...",
        "Recommendation 2...",
        "Recommendation 3..."
    ]
    }
    ```

    Below is rundown of the `risk_pipeline.py` trigggered and used in this endpoint:
    1. Public facade: 
        - `run_full_analysis()` is the function FastAPI calls.
        - It receives key inputs (such as user ID, forecast start date, horizon length) and just orchestrates two big helpers:
            - _fit_forecast_models(): builds & trains the forecasting models.
            - _run_mc_simulation(): turns those forecasts into thousands of balance paths, risk metrics and advice.
    2. Data loading & feature engineering: 
        - `_fit_forecast_models()` starts by reading `synthetic_transactions_daily_enriched.csv` and filtering the chosen user.
        - For every historic row, it adds calendar flags:
            - day_of_week (0–6)
            - month (1–12)
            - is_first_of_month
            - Dutch public-holiday flag

        Income and expense series are then scaled to the 0-1 range so the neural nets train faster.
    3. Training samples creation: A sliding-window generator slices one year of history (L = 365) plus the entire forecast horizon (H = 92 by default) into supervised samples:
        - Encoder input (Xe): last 365 days of scaled income, expenses and calendar flags.
        - Decoder input (Xd): future calendar flags only.
        - Targets:
            - Binary “salary coming?” flag (one per future day).
            - Income amount quantiles.
            - Expense amount quantiles.

        Windows ending in 2024 are used for training; those in 2025 form a small validation set.

    4. Neural-network architecture: `_build_transformer()` builds a tiny Transformer with positional encoding, multi-head attention, layer-norm, dropout, and a feed-forward block. It is reused three times with different output layers:

        - Classifier: predicts a probability that salary appears each day (sigmoid output).
        - Income regressor: predicts p10 / p50 / p90 of salary size (masked pinball loss so it only scores days where salary is present).
        - Expense regressor: predicts p10 / p50 / p90 of total expenses (pinball loss). 

        Each model is trained for up to 50 epochs with Adam and early-stopping on the small validation set. 


    5. Generating the deterministic forecast: Once trained, the three models receive one year of recent history (encoder) and the next H calendar descriptors (decoder). They output, for every forecast day:
        - Probability of a salary event.
        - Salary amount p10/p50/p90.
        - Expense amount p10/p50/p90.

        After rescaling back to euros the forecasts are saved to `final_income_expense_predictions.csv` and stored in a pandas DataFrame (`pred_df`) handed to the Monte-Carlo stage.

    6. Monte-Carlo balance simulation: `_run_mc_simulation()` starts by rebuilding the user’s historical balance curve (cumulative net-cash-flow) and remembers the closing figure as the initial balance. It then runs N = 10 000 independent scenarios:
        - For each future day pick a random salary and expense within the forecast p10–p90 band.
        - Add the net flow to yesterday’s balance and store the result. 

        uring the loop the progress callback is called roughly every 10 % so long simulations don’t feel frozen.

    7. Risk metrics & insights: From the 10 k paths the pipeline provides:
        - Percentile fan-chart (median, p5, p95)
        - Best and worst paths (those ending highest / lowest)
        - VaR & CVaR at 95 % and 99 % confidence
        - Probability of overdraft
        - Simple rule-based text: an intro paragraph, a headline, supporting sentence and up to three actionable bullet recommendations. 

    8. Final JSON: Everything from history, forecast bands, paths, insights, recommendations and extra metrics are then returned as one JSON-serialisable dictionary that main.py sends to the app. 


*Note:* All arrays for balances align with the `dates` array (same length = horizon). The `hist_dates` and `hist_balances` give the tail end of historical trend (for example, the last 90 days before start) so the client can plot them before the forecast. The `median`, `p5`, and `p95` arrays form the main fan chart data (e.g., on 2025-06-01, median balance might be €10,000, p5 might be €5,000, p95 €15,000, meaning 90% of scenarios end between 5k and 15k). The `best` and `worst` arrays represent actual scenario paths from the simulation that achieved extreme outcomes (these can be plotted as dashed lines to show volatility). The `insights` list contains 1-3 key insights in text form (they may include Markdown-style bold highlights, which the app will render accordingly), and `recommendations` is a list of bullet-point advice strings.



- `POST /download_risk_analysis/{user_id}`: Downloads the full risk analysis report as a ZIP. This endpoint returns the results in a file format.
    - Path Parameters: user_id (user to analyze).
    - Query Parameters: start (date), horizon (days) (same meaning as above).
    - Response: A binary response containing a ZIP file. The ZIP (risk_analysis_<user_id>.zip) contains:
        - `forecast_data.csv`: a CSV file with all the raw simulation results. This data allows offline analysis or verification of the simulation.
        - `balance_forecast.png`: an image file of a chart. This is a rendered graph of the fan chart, similar to what the app shows. 

    The Flutter app uses this endpoint when the user taps "Download". The app will handle the file download and storage via its platform-specific code (the implementation uses a helper to save the bytes to a file). From a user perspective, this gives them a way to export their analysis results. However, this should still be further developed and changed. 

Note: Make sure that when you are downloading all the python files and once you have them open, you also adjust the directories to where you have saved the file. For example, in `main.py`, change the following code to where you have saved the futureguard_api folder: 

``` python 
BASE_DIR = Path(
    r"C:/Users/loics/OneDrive/Documents/1. BAM/BLOCK 5/Assignment coding"
)
```

Additonally, there is no authentication on these endpoints (since it’s an internal MVP). In a production setting, these would be protected (e.g., requiring an auth token or being served within an authenticated session in the mobile app). For now, any user ID can be requested freely (the assumption in testing is you only run the app locally and control both front and back ends).

## Frontend Features and User Flow

The FutureGuard Flutter app presents a neat and intuitive user interface to interact with the risk analysis system. Following is a brief description of the main features of the UI and the overall user interaction flow:

1. User Selection & Overview (Home Tab): The app is in the Overview tab when opened. On top are controls to allow the user to select:
    - User ID: There is a dropdown to toggle between various user profiles (here, between 1–100). Since it's a prototype, this is simulating selecting a logged-in user's account. In an actual app, the user would be implicitly known once logged in, but here you can toggle to view various synthetic profiles.    
    - Start Date: A date picker button displays the current start date to examine (default may be current date or recent date). The user may specify an alternative date, which is really saying "I'd like to view my data up to and including this date and project thereafter." It might be utilized in viewing what the analysis would have looked like some time ago or simply for managing the snapshot of transactions displayed.
    - Horizon: Drop-down menu for forecast horizon in days (i.e., 30, 61, 92, 183, 365 days). This specifies how far into the future the risk analysis will forecast.
    
    On changing any of these, the app retrieves the account overview for the given parameters automatically (through OverviewState.fetch() invoking the API). The Overview page then renders:

    - Account Header: A card displaying the user's Name, IBAN, and opening Balance (balance at starting date). This provides background information regarding the user's funds at the beginning.
    - Suggested Action icons: Four colored icons for suggested actions or settings(the "widgets" of the API). In this MVP, they are dummy items that do not do anything, but they illustrate how a bank app typically appears (with much inspiration from the ING app).
    - Recent Transactions List: A list of the most recent transactions until the start date. Each entry displays the date, counterparty, and amount (with +/- or color coding for debit/credit). This provides the users with a recognizable transaction history presentation. It's done using a custom TxTile widget for styling consistency.

    The layout of this page uses Material Design defaults, with a light card for the header and action chips. The user can navigate transactions to see their previous spending.

2. Navigating to Risk Analysis (Risk Tab): In the bottom navigation bar, there's a "Risk" tab (with an icon of a chart/trending-up). Upon tapping this, you're navigated to the Risk Dashboard screen. Following is the standard interaction on this screen:
    - If the user hasn't analyzed yet for the current selection, they are presented with a centered prompt. A big "Run analysis" button (with icon) welcomes them to initiate the risk analysis. This state is shown initially or whenever a new user or date range is chosen and no analysis has been conducted yet.
    - The moment the user clicks on "Run analysis", the app gives instant feedback: it displays a message such as "We are working on your personalized AI-driven Risk Analysis. We will let you know once analysis is ready." and a loading spinner appears on the screen. This conveys the fact that the request is being processed. The UI is programmed to manage the waiting time by disabling the run button and displaying the spinner/text.
    - Once the backend has responded briefly (assuming it returns), the app receives the RiskResult and updates the state. It shows a confirmation message that says "Risk analysis has been finalized!", then the dashboard's content appears (instead of the spinner).

3. Displaying the Risk Dashboard: With results present, the Risk tab is now a scrollable dashboard of data:

    - A refresh button and "Run analysis again" is present at the top. The user can re-run the analysis as required (e.g., if they have changed some scenario parameters).
    - And then there is the Fan Chart. The fan chart is interactive: hover or tap can show tooltips of the precise values (the `fl_chart` library does support displaying a dot and label on touch events, which we activate in the chart configuration). This allows users to obtain precise figures for particular dates if required.
    - The fan chart must represent two areas clearly: the historical actual balance (plotted up to the start date) and the forecast area. E.g., historical in blue, forecast median in orange with orange shaded area for 5th and 95th percentile range. This is the standard form of financial forecast graphs (sometimes also referred to as "fan charts" in economics).
    - Below the chart, the Insights card is displayed. It is a Material card with the title "Insights". There is a new line for each insight and it is centered. The insights place things in perspective like worst-case scenarios for instance and the Value at Risk.
    - After insights, the Recommendations card is displayed (title "Recommendations"). This is a bulleted list of up to 3 recommendations. These are usually action-oriented (e.g., "Reduce monthly expenses by 9% to decelerate the deficit"). They are created by backend simple rules.
    - Finally, the Download ZIP button is displayed (centered). This button allows the user to download the analysis report in the two files discussed (CSV + PNG files).

    All this inside a ScrollView so that the user can scroll if it won't all fit on one screen (particularly on smaller devices).

**User Interaction Flow Summary:**

1. The user opens the app (no login in MVP; default user ID 10 and some date/horizon may be pre-selected by default).
2. On Overview tab, the user is presented with his/her current balance and transactions. He/she can modify the user ID to view someone else's information, or select a different date range. The overview is updated accordingly (a new API call is made every time it is performed).
3. The user goes to Risk tab. Initially, sees the prompt and clicks Run analysis.
4. The app invokes the API and displays a loading indicator. The user waits for approximately 30 minutes.
5. The results are displayed. The user examines the graph of potential futures and notices, for instance, that the shaded area widens over time (to indicate increasing uncertainty). They click at the tip on the graph to find out what the worst-case value could be. The tooltip displays for instance "€-500 at 5%".
6. The user can read the Insights and Recommendations. 
7. The user chooses to click Download analysis in order to retain a copy. The app downloads the zip.
8. Satisfied, the user may go back to Overview or change horizon to see a different forecast. They select say 92 days horizon and run again. The app repeats the process, now showing a different fan chart. 10. The user also has the opportunity to try various user profiles from the dropdown (how it would be with different people who possess different financial profiles). 

*Note on **file technicalities**:* If you're working with VS Code, ensure that when opening a new window, you go to `file` at the top left and then `Open Workspace from File`. Once you have opened the futureguard_app, you can see the workspace in VS Code. Once you have opened the workspace, you can open the terminal and enter `flutter pub get`. Then, execute `flutter run -d chrome` and the app will launch in chrome.

*Note on **State and Navigation**:* The app maintains the selected state when changing tabs. So the user can switch back and forth between Overview and Risk tabs to cross-reference without needing to re-enter selections.
For example, if one sees an enormous transaction cost and wonders what effect it has on the analysis, one can go to Risk (which already has results for the same date range). If they go back and select a different date, the risk result is stale (since it was for an older date) and they'd rerun it for the new context.

**Planned Enhancements**: Although the present UI is usable, some enhancements are planned to be made:

- Incorporating interactive sliders to enable user-specifiable scenario stress tests (for example, a slider to model "income loss %" or "incremental monthly spending"). The architecture already includes placeholders in ScenarioState for parameters such as these. A future release would expose these as user-controllable prior to running the analysis to determine how results vary under user-defined scenarios (for example, simulate a 20% salary reduction and then run analysis). 
- A separate Settings or More tab (third tab is a placeholder currently) for account settings, risk tolerance toggle, or changing assumptions. 
- Polished UI/UX: Theming the design to the bank, including animations (perhaps an animation during simulations run or a prettier transition to results), etc. 
- Tooltips and Info Modals: Display definitions for terms such as VaR or Monte Carlo when the user clicks on an info icon. So far, the intended audience (investor or tech reviewer) is assumed to be knowledgeable but end-users may require onboarding. 

The Flutter frontend demonstrates how to present complex data (the results of thousands of simulations) in an interpretable and concise form. Graphs, together with concise text-based conclusions, make probabilistic results easy to understand, fulfilling the project mission to make financial risk easier to grasp for users.

## Deployment Guide
FutureGuard is now set up for local testing and deployment today. Both backend and frontend run on a developer's machine. There hasn't been full production or cloud deployment yet, but the following are some notes regarding local deployment and also some thoughts regarding future cloud deployment:

**Local Deployment (Development):**
- Complete the Setup and Installation instructions to have the backend up and running on your computer (for example, at http://127.0.0.1:8000) and the Flutter app up and running (in an emulator or browser). 
- Set the API base URL in the Flutter application to your local backend. If you are using an Android emulator, use 10.0.2.2 as indicated. If you are using a browser, you may need to enable CORS on the FastAPI application (FastAPI's CORSMiddleware) because browsers implement CORS – but if you are using Chrome with flutter run -d chrome, you may need to permit localhost or run with a flag. During development, you can set FastAPI to enable http://127.0.0.1:* origins.
- Run the backend using uvicorn (you can leave --reload for auto-reload while developing). Leave that window open to see logs. It will log requests to endpoints, which helps with debugging.
- Run the Flutter application in debug mode. If you are using VS Code or Android Studio, you can insert breakpoints or print logs in Dart to debug UI flow or API responses.
- End-to-end test the process: pick a user, get overview, execute analysis, view results. Monitor the terminal where the backend is running to see if the calls were attempted and finished (it will display HTTP 200 for called endpoints, or errors/exceptions).

**Cloud/Server Deployment Recommendations:** (Not installed in this project, but future recommendations)
- Replace CSVs with a real live dynamic database for long-term data if moving to production.
- The Flutter app would be published via the App Store/Play Store by the host bank app.
- For Networking & Security, the backend must be behind HTTPS (TLS) and possibly behind an API gateway. The application would require the domain of the API (e.g., api.futureguard.com) rather than localhost. 
- Scaling: As FastAPI is asynchronous, it can manage numerous requests, yet for CPU-bound tasks (such as executing the trasnformers model or the thousands of simulations), one would need to scale out the service.
- Cloud deployment is not part of the present state. This would be set up as a future task as an MVP-to-live-pilot handover. 

For the time being, any developer or reviewer can test FutureGuard on their local machine by following the steps above. The priority was to build the functionality and not the deployment scripts that would be included once the project would have moved beyond MVP phase.


## Security Considerations

As a prototype, FutureGuard has minimal security in place, but it’s important to note best practices and what should be addressed before any real-world deployment: 

- **Authentication & Authorization**: Authentication is not implemented in the present API. In fact, the user in the example above can request any user's information by ID. This was left for demo purposes. In a production application, this would not be acceptable. We would have used the bank authentication system (i.e., OAuth2 or JWT tokens in headers) to make sure the requesting user only receives their own information. Every endpoint would verify the identity and permissions of the user (for instance, `/account_overview/{user}` would validate session user is `{user}`).
- **Data Privacy**: Personal financial transactional data is highly sensitive. Although our dataset is fictional, in the real world all personal data needs to be processed according to GDPR and banking data regulations. That implies:
    - Do not reveal more information than required in API responses.
    - Possibly encrypt sensitive information at rest (in case transactions or model outputs are being stored). MVP stores in CSV, but a production database would include access controls and encryption.
- **Rate Limiting & Performance**: An attacker or user might spam the `/run_risk_analysis` endpoint which is resource-intensive. We should rate limit (application or API gateway level) to avoid denial of service or overloading. (e.g., one analysis per user at a time, or X per minute).
- **Model Security**: ML models can be vulnerable to exploitation (by crafted input causing unexpected behavior) from time to time. Our input here is numerical and small, so not a large attack surface. However, we do want to make sure the model will perform nicely with edge cases (e.g., all zeros inputs, extreme values) without crashing or outputting very bad results.
- **Dependency Security**: We depend on third-party libraries. It's a good idea to keep up to date in order to receive security fixes. If, for example, a vulnerability were discovered in FastAPI or TensorFlow, we would want to upgrade. Using a tool to check for dependencies (such as `pip-audit` or dependabot on GitHub) is best practice.
- **Frontend Security**: As the Flutter application is present on the users' devices, it can be examined. Although our application does not contain any secrets (no API keys or hardcoded credentials), in an integrated real-world scenario we would use secure storage to store auth tokens. Refrain from logging sensitive information in the app.
- **Auditability**: As a financial instrument, it may be significant to log what analysis was executed and when. In a future release, one would incorporate logging on events such as "User X executed analysis on date Y with horizon Z". This would be useful for debugging purposes as well as in any user questions or compliance audits. 

In summary, though the MVP isn't nailed down for public use, it's designed in such a manner that incorporating security doesn't prove to be challenging (courtesy of FastAPI's design and Flutter's versatility). The aforementioned measures need to be taken before any actual user data is utilized to ensure user privacy and gain trust.



## Scaling and Maintenance Notes
As FutureGuard transitions from MVP to a production system, there are several considerations for scaling the system and maintaining it over time:
- **Handling More Data & Users**: The synthetic dataset is relatively small (100 users, ~3.5 years of data). In a real scenario with potentially millions of users and much longer histories, the data processing and storage needs become larger:
    - Move from CSV files to a robust **database**. A relational DB or time-series database can handle transactions and balances efficiently. 
    - The ML model might need to be trained on a representative dataset and then possibly fine-tuned per user. We may not train a unique model for each user (that’s not scalable), but rather use one global model or a model per segment of users. Maintenance could involve periodic retraining as more data comes in (to avoid model drift).
- **Scaling the Backend Compute**: Time Series Transformers and Monte Carlo simulations can be CPU-intensive if horizon or scenario count is large. One strategy to limit issues could be by doing **Batch computations**. Indeed, in a bank scenario, some computations could be done offline. For example, the risk analysis could be precomputed nightly for all users as a batch job, and then the app just fetches stored results which is instant.
- **Maintaining the ML Model**: Over time, as user behavior changes or new economic conditions arise, the model should be adjusted. Also consider expanding the model to include more features or using more advanced architectures if needed. This could be tested in the notebooks first for example.
- **Feature Updates and UI Scaling**: Adding new features (like the stress-test sliders, or additional dashboard elements) should be done carefully to keep the UI intuitive. 
- **Logging and Monitoring**: In production, implement monitoring for both backend and app:
    - Use logging (with structured logs) in FastAPI to track performance of each request (time taken, etc.) and any errors.
    - Use an Application Performance Monitoring tool to watch the system’s health.
    - Crash reporting for the Flutter app (Firebase Crashlytics or Sentry) to catch any runtime errors users face.
    - Analytics to see how users are interacting (do they often not click download? Do they run analysis multiple times? etc.), which can guide UX improvements.
- **Maintaining Data Pipeline**: The synthetic data generation isn’t needed in production, but if any part of the feature engineering is ported to real data, maintain those scripts. Possibly create a pipeline to update features as new transactions come in (incremental feature calc rather than full recompute).
- **Extensibility**: For maintainability, keep the code modular: e.g., separate the ML prediction code, simulation code, and recommendation code into different modules or services if it grows. The current single FastAPI app is fine for MVP, but modularization will help multiple developers work on it concurrently without stepping on each other’s toes.
- **User Feedback Loop**: As this is a new kind of feature for users, gathering user feedback and making iterative improvements is key. Maintenance is not just technical upkeep, but also ensuring the tool remains helpful. For instance, if users are confused by VaR, perhaps we add more explanation in-app. If a recommendation is ignored by 100% of users, maybe it’s not useful and should be changed.

In essence, FutureGuard is built with scalability in mind (stateless backend, separate frontend), so scaling out horizontally is straightforward. The main heavy lifting is the ML and simulation – with proper optimization and possibly upgrading hardware (using cloud instances with more CPU or using vectorized libraries), the app can scale to handle many users. Maintenance will involve periodic updates to models and data, as well as codebase improvements as the code transitions from prototype quality to production quality (adding documentation, tests, and refactoring where necessary).



