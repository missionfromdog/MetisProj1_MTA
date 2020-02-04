# Project Benson Team Repository | Metis Winter 2019
## By [Andy](https://github.com/4thLawOfRobotics), [Blake](https://github.com/ChicagoDev), and [Casey](https://github.com/missionfromdog)

### Project Summary

​	Project Benson is a set of tools for analysing the NYC MTA subway ridership.  

### Project File Descriptions

#### **MtaRidership.py**

Simplifies loading, parsing, and accessing MTA data. Creates our MVP: a bar chart with overall ridership volume. MVP is paramaterized.

Other miscelaneous functions include interquartile range and a ridership share percentage function.   


#### **ABC_ANALYTICS_PLOTLY_MTA.ipynb**

  A Jupyter notebook that takes the Stations CSV, uses the [EPSG](http://spatialreference.org/ref/epsg/nad83-new-york-long-island-ftus/) for Spatial Reference  to grab NYC.  It then uses Plotly to post stations via Lat/Lon.


#### **TurnstileData.ipynb**

A Jupyter notebook that loads the MtaRidership class and all 2018 turnstile data. This notebook shows how we refined our initial code before putting it into the class.

You can download the compressed Pickle from [Google Drive](https://drive.google.com/open?id=1Y8tN1pdgTk9SD0lOeNj4l6sA6AI06abK).
This will allow you to run all the cells in the notebook.

We also used this notebook to produce some of the graphs for your presentation.
