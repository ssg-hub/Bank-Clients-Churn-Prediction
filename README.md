# Bank-Clients-Churn-Prediction
Deploying on Streamlit, an app to analyse the classification of credit card churners.

# learrning project

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![MIT License][license-shield]][license-url]

<!-- PROJECT TITLE -->
<br />
<h2 align="center">Churn Prediction - Classification Data Analysis Project with Streamlit</h2>
<p align="center"><a href="https://github.com/ssg-hub/Churn-Prediction-project">
<img src="https://github.com/ssg-hub/ImmoEliza-Regression-project/blob/main/logo_Bouman_3.31.png" alt="Logo" width="200" height="200"></a></p>
<h3 align="center">API Deployment Project to get predictions for prices of properties in Belgium based on property data input, as a part of Data&AI training at <a href="https://github.com/becodeorg"><strong>BeCode</strong></a></h3><br><br>

<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#installation">Installation</a>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#resources">Resources</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About The Project

This Machine Learning API deployment project was a part in the intermediate phase of Data and AI projects at Becode, Belgium.
The data and the format of the data required to be input by the user is found at [ancient-journey-94670.herokuapp.com/predict/](ancient-journey-94670.herokuapp.com/predict/). The required property parameters are the _zipcode, area, type, number of rooms_ and presence of these factors _kitchen-equipped, furnished, fireplace, terrace, garden, swimming-pool and building-condition_.


**The key result of this application in the _price of the property_  in Euros, predicted using the required parameters of the property to buy in Belgium.**
This prediction is done using a machine learning Gradient Boost regression model built using a dataset, that was scraped from [www.immoweb.be](www.immoweb.be) in an earlier project.

### Built With

* [Python 3.8.10](https://www.python.org/)

<!-- GETTING STARTED -->

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/ssg-hub/ImmoEliza-API-Deployment.git
   ```

2. Install dependencies
   ```sh
   pip install -r requirements.txt
   ```

<!-- USAGE EXAMPLES -->
## Usage

Run the app on your local machine
   ```sh
   $ python3 streamlit_classifier.py 
   ```

<!-- CONTRIBUTING -->
## Contributing

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/NewFeature`)
3. Commit your Changes (`git commit -m 'Add some NewFeature'`)
4. Push to the Branch (`git push origin feature/NewFeature`)
5. Open a Pull Request

<!-- RESOURCES -->
## Resources
[Classification](https://scikit-learn.org/stable/index.html) , [Streamlit](https://docs.streamlit.io/en/stable/getting_started.html), [Build Docker images](https://devcenter.heroku.com/articles/build-docker-images-heroku-yml#getting-started), [Deploying on Heroku](https://devcenter.heroku.com/articles/git#tracking-your-app-in-git)

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.

<!-- CONTACT -->
## Contact
Shilpa Singhal 

Project Link: [https://github.com/ssg-hub/ImmoEliza-API-Deployment](https://github.com/ssg-hub/ImmoEliza-API-Deployment)

