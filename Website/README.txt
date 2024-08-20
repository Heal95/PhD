Dubrovnik shaking scenarios website - https://dubrovnikshakingscenarios.gfz.hr
------------------------------------------------------------------------------
The website consists of several static files that need to be served through a server program (e.g., NGINX) to function correctly:

● index.html: Defines the components on the page.
● index.css: Defines the style of the components.
● index.js: Contains the logic for displaying data.
● favicon.png: The website icon in PNG format.
● favicon.svg: The website icon in SVG format.
● ol-v4.6.5.css: The OpenLayers library for map display styling.
● ol-v4.6.5.js: The OpenLayers library for map display logic.
● Znak_Zavoda_kvadrat.png: Image of the institute's logo.
These static files need to be placed in a main working directory that will be served through the selected server program.

The results displayed on the website need to be organized as follows:
● Images of the available scenarios should be placed in a subdirectory named ‘scenariji’.
● CSV data should be placed in a subdirectory named ‘podaci’.

The data must consist of the following files and subdirectories:
● podaci/DATA.zip: A file containing all the data.
● podaci/STATIONS.txt: A list of data points.
● podaci/files/: A directory with data for each data point.
● podaci/figs/E/: A directory with images of the 'E' component for each data point.
● podaci/figs/N/: A directory with images of the 'N' component for each data point.
● podaci/figs/Z/: A directory with images of the 'Z' component for each data point.
The data must be prepared for use by running the script ‘prepare.sh’. This bash script must be placed in the ‘podaci’ directory before running. The script will generate the files necessary for loading the data on the page. Figures were plotted using plot_data.py.
