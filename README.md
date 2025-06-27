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

The FutureGuard Flutter app provides a simple and clean user interface to interact with the risk analysis system. Here is an outline of the main features of the UI and the typical user interaction flow: 

1. User Selection & Overview (Home Tab): When the app starts, it opens on the Overview tab. At the top, there are controls for the user to select:
    - User ID: A dropdown menu allows switching between different user profiles (here between 1–100). Since this is a prototype, this simulates selecting a logged-in user’s account. In a real app, the user would be implicitly known after login, but here you can toggle to see different synthetic profiles.
    - Start Date: A date picker button shows the current start date for analysis (default might be the current date or a recent date). The user can pick a different date, which essentially means “I want to see my data up to this date, and forecast after it.” This could be used to see what the analysis looked like at a past point in time or simply to control the snapshot of transactions shown.
    - Horizon: A dropdown for forecast horizon in days (options like 30, 61, 92, 183, 365 days). This determines how far ahead the risk analysis will predict into the future.

    After adjusting any of these, the app automatically fetches the account overview for the selected parameters (via OverviewState.fetch() calling the API). The Overview page then displays:
    - Account Header: A card showing the user’s Name, IBAN, and current Balance (balance as of the start date). This gives context about the user’s finances at the starting point.
    - Suggested Action icons: Four colored icons representing suggested actions or settings(the “widgets” from the API). In this MVP, they are non-interactive placeholders, but they demonstrate how a bank app usually looks (with great inspiration from the ING app).
    - Recent Transactions List: A scrollable list of the latest transactions up to the start date. Each entry shows the date, counterparty, and amount (with color coding or +/- for credit/debit). This gives users a familiar transaction history view. It’s implemented with a custom TxTile widget for consistency in style. 

    The design of this page follows Material Design defaults, with a light card for the header and chips for actions. The user can scroll through transactions to review their past spending. 

2. Navigating to Risk Analysis (Risk Tab): The bottom navigation bar has a "Risk" tab (with an icon of a chart/trending-up). When the user taps this, it switches to the Risk Dashboard screen. Here’s the typical interaction on this screen:
    - If the user has not yet run an analysis for the current selection, they will see a centered prompt. There is a large “Run analysis” button (with an icon) inviting them to start the risk analysis. This state occurs initially or whenever a new user or date range is selected and no analysis has been done yet.
    - When the user taps "Run analysis", the app immediately provides feedback: it shows a  message like “We are working on your personalized AI-driven Risk Analysis. We will let you know once analysis is ready.” and a loading spinner appears on the screen. This indicates that the request is in progress. The UI is designed to handle the waiting period by disabling the run button and showing the spinner/text.
    - After a short time (assuming the backend responds), the app receives the RiskResult and updates the state. A confirmation notification is shown saying “Risk analysis has been finalized!”, and then the dashboard content becomes visible (replacing the spinner).

3. Viewing the Risk Dashboard: With results available, the Risk tab becomes a scrollable dashboard of information:
    - At the top, a refresh icon and “Run analysis again” button is available. The user can re-run the analysis on demand (for example, if they have adjusted any scenario parameters).
    - Next is the Fan Chart. The fan chart is interactive: hovering or tapping can display tooltips of the exact values (the `fl_chart` library supports showing a dot and label on touch events, which is enabled in the chart configuration). This helps users get exact numbers for specific dates if needed.
    - The fan chart clearly shows two regions: the historical actual balance (plotted up to the start date) and the forecast region. For instance, historical is plotted in blue, and the forecast median in orange with the orange shaded band for 5th and 95th percentile range. This visual design aligns with typical financial forecast charts (sometimes called "fan charts" in economics).
    - Just below the chart, the Insights card appears. This is a Material card with the header “Insights”. Each insight is on a new line and centered. The insights give context such as worst-case outcomes for example and the Value at Risk.
    - Following insights, the Recommendations card is shown (titled “Recommendations”). This contains a bulleted list of up to 3 recommendations. These recommendations are typically action-oriented (e.g., “Cut monthly spending by 9% to slow the deficit”). They are generated by simple rules in the backend.
    - Finally, the Download ZIP button is displayed (centered). This button allows the user to save the analysis report with the two files aforementioned (CSV + PNG files).

All of this content is within a ScrollView so the user can scroll if it doesn’t fit on one screen (especially on smaller devices). 

**User Interaction Flow Summary:**

1. The user opens the app (no login required in MVP; by default user ID 10 and some date/horizon might be pre-selected).
2. On Overview tab, user reviews their current balance and transactions. They might change the user ID to see another profile’s data, or pick a different date range. The overview updates accordingly (triggering a new API call each time).
3. The user navigates to Risk tab. Initially, sees the prompt and hits Run analysis.
4. The app calls the API and shows a loading state. The user waits for around 30 minutes.
5. The results appear. The user sees the chart of possible futures and notices, for example, that the shaded area widens over time (showing uncertainty growing). They tap on the chart near the end to see what the worst-case value might be. The tooltip shows for example “€-500 at 5%”.
6. The user can read the Insights and Recommendations. 
8. The user decides to tap Download analysis to keep a copy. The app saves the zip file. 
9. Satisfied, the user may go back to Overview or change horizon to see a different forecast. They select say 92 days horizon and run again. The app repeats the process, now showing a different fan chart. 
10. The user can also experiment with different user profiles using the dropdown (simulating how it might work for different individuals with different financial profiles).

*Note on **file technicalities**:* If you are using VS Code, make sure that once you have a new window open, to select `file` on the top left and then `Open Workspace from File`. Once you have selected the futureguard_app, you will be able to see the workspace in VS Code. Once the workspace is open, you can open the terminal and write `flutter pub get` and press enter. Then, write `flutter run -d chrome` and you will have the app open up in chrome. 

*Note on **State and Navigation**:* The app maintains the selected state when switching tabs. So the user can flip between Overview and Risk tabs to cross-check information without re-entering selections. For example, if one sees a large expense in transactions and wonder how it affected the analysis, one can switch to Risk (which is already showing results for that same date range). If they go back and select a different date, the risk result is considered stale (since it was for a previous date) and they’d run it again for the new context. 

**Planned Enhancements**: While the current UI is functional, there are plans to enhance it further:
- Adding interactive sliders for custom scenario stress tests (e.g., a slider to simulate “income loss %” or “extra monthly expense”). The architecture already has placeholders in ScenarioState for such parameters. A future update would allow the user to adjust these before running the analysis to see how the outcomes change under custom scenarios (for instance, simulate a 20% pay cut and then run analysis).
- A dedicated Settings or More tab (the third tab is a placeholder now) for things like account settings, toggling risk tolerance, or adjusting assumptions.
- Polished UI/UX: Aligning the design with the bank’s theme, adding animations (maybe an animation while simulations run or a nicer transition to results), etc.
- Tooltips and Info Modals: Provide explanations for terms like VaR or Monte Carlo if the user taps an info icon. For now, the target audience (investor or tech reviewer) is expected to understand, but end-users might need onboarding.

The Flutter frontend shows how complex data (the result of thousands of simulations) can be presented in a concise and user-friendly way. Charts, combined with short textual insights, make it easier to grasp probabilistic outcomes, fulfilling the project’s aim to improve user understanding of financial risk.

## Deployment Guide
Currently, FutureGuard is configured for local deployment and testing. Both the backend and frontend run on a developer’s machine. A full production or cloud deployment has not been implemented yet, but here are notes on how to deploy locally and considerations for eventual cloud deployment: 

**Local Deployment (Development):**
- Follow the Setup and Installation steps to get the backend running on your machine (e.g., at http://127.0.0.1:8000) and the Flutter app running (either in an emulator or web browser).
- Ensure that the Flutter app’s API base URL is pointing to your local backend. If using an Android emulator, use 10.0.2.2 as noted. If using a web browser, you might need to enable CORS on the FastAPI app (FastAPI’s CORSMiddleware) since browsers enforce CORS – but if you’re using Chrome with flutter run -d chrome, you might need to allow localhost or run with a flag. In development, you can configure FastAPI to allow http://127.0.0.1:*/ origins.
- Run the backend with uvicorn (you can keep --reload for auto-reload during development). Keep that terminal open to see logs – it will log requests to endpoints, which is useful for debugging.
- Run the Flutter app in debug mode. If using VS Code or Android Studio, you can put breakpoints or print logs in Dart to debug UI behavior or API responses.
- Test the flow end-to-end: select a user, get overview, run analysis, see results. Check the terminal running the backend to see that the calls were received and completed (it will show HTTP 200 for endpoints called, or any errors/exceptions).

**Cloud/Server Deployment Recommendations:** (Not configured in this project, but guidelines for future)
- Set up a proper live and dynamic database for persistent data instead of CSVs if moving to production.
- The Flutter app would be distributed via the App Store/Play Store through the host banking app. 
- For Networking & Security, the backend should be behind HTTPS (TLS) and possibly behind an API gateway. The app would need the domain of the API (e.g., api.futureguard.com) instead of localhost.
- Scaling: Since FastAPI is asynchronous, it can handle many requests, but for CPU-heavy tasks (like running the trasnformers model or the thousands of simulations), one might want to scale out the service.
- In the current state, cloud deployment is not yet integrated . Setting this up would be a next step when transitioning from MVP to a live pilot.

For now, any reviewer or developer can run FutureGuard locally using the instructions above. The focus was on building the functionality and not on deployment scripts that would be added once the project would move past MVP stage.

## Security Considerations

As a prototype, FutureGuard has minimal security in place, but it’s important to note best practices and what should be addressed before any real-world deployment:
- **Authentication & Authorization**: The current API does not enforce any authentication. Indeed, the user here can query any user’s data by ID. This was made for presentation purposes. In a real world application, this would be unacceptable. We would integrate with the bank’s authentication system (e.g., OAuth2 or JWT tokens in headers) to ensure the requesting user only accesses their own data. Each endpoint would validate the user’s identity and permissions (for example, `/account_overview/{user}` should ensure the session user matches `{user}`).
- **Data Privacy**: Financial transaction data is highly sensitive. Even though our dataset is synthetic, in a real scenario all personal data must be handled according to GDPR and banking data standards. This means:
    - Never expose more data than necessary in API responses.
    - Possibly encrypt sensitive data at rest (if storing transactions or model outputs). The MVP stores in CSV. However, a production database should use encryption and access controls.
- **Rate Limiting & Performance**: An attacker or even an overly keen user could spam the `/run_risk_analysis` endpoint which is computationally expensive. We should implement rate limiting (at API gateway or app level) to prevent denial of service or excessive load (e.g., limit to one analysis at a time per user, or X per minute). 
- **Model Security**: Machine learning models can sometimes be exploited (through malicious input causing unexpected behavior). Here our input is limited and numeric, so not a big attack surface. But we should still ensure the model can handle edge cases (like all zero inputs, extreme values) without crashing or providing really bad results. 
- **Dependency Security**: We rely on third-party libraries. It’s important to keep them updated to get security patches. For instance, if a vulnerability is found in FastAPI or TensorFlow, we’d need to update. Using a tool to monitor dependencies (like `pip-audit` or GitHub’s dependabot) is recommended.
- **Frontend Security**: The Flutter app being on user devices means it could be inspected. While our app has no secrets (no API keys or hardcoded credentials), in a real integrated scenario we might use secure storage for auth tokens. Ensure not to log sensitive info in the app.
- **Auditability**: For a financial tool, logging what analysis was done and when could be important. In a next iteration, one would add logging for actions like “User X ran analysis on date Y with horizon Z”. This would help in debugging and also in any user inquiries or compliance checks.

In summary, while the MVP is not secured for public use, it’s built in a way that adding security is straightforward (thanks to FastAPI’s design and Flutter’s flexibility). Before any real user data is used, the above measures must be implemented to protect user privacy and maintain trust.

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

