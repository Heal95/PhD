const scenarios = {
    "scenario-1": {
        name: "Scenarij 1",
        image: "scenariji/Scenario1.png",
        videoUrl: "https://drive.google.com/file/d/1FX5IIs0T1K9DuRE0kgNij4_O8jZTR4PB/view?usp=sharing",
        data: "podaci_link/Scenario1/Scenario1_data.zip",
        imagesDir: "podaci_link/Scenario1",
        points: [], // set in different file
        initiationPoint: {
            latitude: "42.619 °N",
            longitude: "18.138 °E"
        },
        numberOfPoints: "916",
        ruptureVelocity: "2.8",
        surfaceArea: "748",
        magnitude: "6.92",
        seismicMoment: "3.02⨯1026",
        focalMechanism: "298°/33°/90°",
        displacementDistribution: "Nejednolika",
        timeSymmetry: "Bilateralna"
    },
    "scenario-2": {
        name: "Scenarij 2",
        image: "scenariji/Scenario2.png",
        videoUrl: "https://drive.google.com/file/d/1ujVaDhuVp_uXBzPFXYtvsb2ZwVEz5sx0/view",
        data: "podaci_link/Scenario2/Scenario2_data.zip",
        imagesDir: "podaci_link/Scenario2",
        points: [], // set in different file
        initiationPoint: {
            latitude: "42.563 °N",
            longitude: "18.312 °E"
        },
        numberOfPoints: "910",
        ruptureVelocity: "2.8",
        surfaceArea: "748",
        magnitude: "6.92",
        seismicMoment: "3.02⨯1026",
        focalMechanism: "298°/33°/90°",
        displacementDistribution: "Nejednolika",
        timeSymmetry: "Unilateralna (jugoistok-sjeverozapad)"
    },
    "scenario-3": {
        name: "Scenarij 3",
        image: "scenariji/Scenario3.png",
        videoUrl: "https://drive.google.com/file/d/1Trbjyc0Q5RslHpm8FAYC_y5D0l40SAKS/view",
        data: "podaci_link/Scenario3/Scenario3_data.zip",
        imagesDir: "podaci_link/Scenario3",
        points: [], // set in different file
        initiationPoint: {
            latitude: "42.711 °N",
            longitude: "17.975 °E"
        },
        numberOfPoints: "920",
        ruptureVelocity: "2.8",
        surfaceArea: "748",
        magnitude: "6.92",
        seismicMoment: "3.02⨯1026",
        focalMechanism: "298°/33°/90°",
        displacementDistribution: "Nejednolika",
        timeSymmetry: "Unilateralna (sjeverozapad-jugoistok)"
    },
    "scenario-4": {
        name: "Scenarij 4",
        image: "scenariji/Scenario4.png",
        videoUrl: "https://drive.google.com/file/d/1sHAxEpkB_3Ekm198eaUW9-SdcWXPfzo0/view",
        data: "podaci_link/Scenario4/Scenario4_data.zip",
        imagesDir: "podaci_link/Scenario4",
        points: [], // set in different file
        initiationPoint: {
            latitude: "42.619 °N",
            longitude: "18.138 °E"
        },
        numberOfPoints: "916",
        ruptureVelocity: "2.8",
        surfaceArea: "748",
        magnitude: "6.92",
        seismicMoment: "3.02⨯1026",
        focalMechanism: "298°/33°/90°",
        displacementDistribution: "Jednolika",
        timeSymmetry: "Bilateralna"
    },
    "scenario-5": {
        name: "Scenarij 5",
        image: "scenariji/Scenario5.png",
        videoUrl: "https://drive.google.com/file/d/1lKpXttzfJk6rnnDjX6Svix0uFvHCBL_K/view",
        data: "podaci_link/Scenario5/Scenario5_data.zip",
        imagesDir: "podaci_link/Scenario5",
        points: [], // set in different file
        initiationPoint: {
            latitude: "42.563 °N",
            longitude: "18.312 °E"
        },
        numberOfPoints: "910",
        ruptureVelocity: "2.8",
        surfaceArea: "748",
        magnitude: "6.92",
        seismicMoment: "3.02⨯1026",
        focalMechanism: "298°/33°/90°",
        displacementDistribution: "Jednolika",
        timeSymmetry: "Unilateralna (jugoistok-sjeverozapad)"
    },
    "scenario-6": {
        name: "Scenarij 6",
        image: "scenariji/Scenario6.png",
        videoUrl: "https://drive.google.com/file/d/1SSc-OKceHvTVU1lEbQQid0_9CNx95QCJ/view",
        data: "podaci_link/Scenario6/Scenario6_data.zip",
        imagesDir: "podaci_link/Scenario6",
        points: [], // set in different file
        initiationPoint: {
            latitude: "42.711 °N",
            longitude: "17.975 °E"
        },
        numberOfPoints: "920",
        ruptureVelocity: "2.8",
        surfaceArea: "748",
        magnitude: "6.92",
        seismicMoment: "3.02⨯1026",
        focalMechanism: "298°/33°/90°",
        displacementDistribution: "Jednolika",
        timeSymmetry: "Unilateralna (sjeverozapad-jugoistok)"
    }
};

const languages = {
    "hr": null,
    "en": {
        "subtitlea": "The great 1667 Dubrovnik earthquake seismic shaking scenarios",
        "scenario-1-name": "Scenario 1",
        "scenario-2-name": "Scenario 2",
        "scenario-3-name": "Scenario 3",
        "scenario-4-name": "Scenario 4",
        "scenario-5-name": "Scenario 5",
        "scenario-6-name": "Scenario 6",
        "description": "Choose a scenario and click on the unshaded part of the map " +
            "to display ground motion velocity at a point.",
        "subtitleb": "Finite-fault model parametrization for the 1667 Dubrovnik earthquake",
        "dropmenu": "Choose a scenario...",
        "initpoint": "Initiation point:",
        "numpoint": "Number of points:",
        "rupvel": "Rupture velocity:",
        "faultarea": "Fault surface:",
        "moments": "Magnitude and seismic moment:",
        "focal": "Focal mechanism:",
        "slip": "Slip distribution:",
        "time": "Time symmetry:",
        "downloada": "Download whole dataset",
        "vidsim": "Video simulation",
        "infoa": "General information",
        "infob": "General information",
        "infoc": "This application enables the reading of low-frequency simulated ground " +
            "motion records for various earthquake shaking scenarios of the 1667 great Dubrovnik " +
            "earthquake. The displayed data is also the result of a doctoral dissertation titled" +
            " \"Seismic shaking scenarios for the wider Dubrovnik area\" by Helena Latečki.",
        "infod": "Individual records and the entire dataset by scenario are available " +
            "for download in text (txt) format.",
        "infoe": "For any questions and additional information, please send an email to: " +
            "helena.latecki[at]gmail.com",
        "ada": "Department of Geophysics",
        "adb": "Faculty of Science, University of Zagreb",
        "adc": "Horvatovac 95",
        "add": "Zagreb, Croatia",
        "infof": "Close",
        "coords": "Coordinates:",
        "comp": "Component:",
        "downloadb": "Download data",
        "popa": "Close",
        "nodata": "There is no data for this point.",
        "popb": "Close"
    }
};

const langReplacements = {
    "en": {
        "Scenarij 1": "Scenario 1",
        "Scenarij 2": "Scenario 2",
        "Scenarij 3": "Scenario 3",
        "Scenarij 4": "Scenario 4",
        "Scenarij 5": "Scenario 5",
        "Scenarij 6": "Scenario 6",
        "Jednolika": "Uniform",
        "Bilateralna": "Bilateral",
        "Nejednolika": "Nonuniform",
        "Unilateralna (jugoistok-sjeverozapad)": "Unilataral (southeast-northwest)",
        "Unilateralna (sjeverozapad-jugoistok)": "Unilateral (northwest-southeast)"
    },
    "hr": {
        "Scenario 1": "Scenarij 1",
        "Scenario 2": "Scenarij 2",
        "Scenario 3": "Scenarij 3",
        "Scenario 4": "Scenarij 4",
        "Scenario 5": "Scenarij 5",
        "Scenario 6": "Scenarij 6",
        "Uniform": "Jednolika",
        "Bilateral": "Bilateralna",
        "Nonuniform": "Nejednolika",
        "Unilataral (southeast-northwest)": "Unilateralna (jugoistok-sjeverozapad)",
        "Unilateral (northwest-southeast)": "Unilateralna (sjeverozapad-jugoistok)"
    }
};

let lang = "hr";

const langHrButton = document.getElementById("lang-hr-button");
const langEnButton = document.getElementById("lang-en-button");

function translateLang(language) {
    lang = language;

    langHrButton.classList.remove("selected");
    langEnButton.classList.remove("selected");

    if (language === "hr") {
        langHrButton.classList.add("selected");
    } else if (language === "en") {
        langEnButton.classList.add("selected");
    }

    let updateHr = false;

    if (!languages["hr"]) {
        languages["hr"] = {};
        updateHr = true;
    }

    Array.from(document.getElementsByClassName("i18n")).forEach(e => {
        const i18nKey = e.getAttribute("data-i18n-key");
        const i18nReplace = e.hasAttribute("data-i18n-replace")

        if (i18nReplace) {
            e.innerText = langReplacements[language][e.innerText] || e.innerText;
        } else if (!i18nKey) {
            console.error("Missing i18n key for element: ", e);
        } else {
            if (updateHr) {
                languages["hr"][i18nKey] = e.innerText;
            }

            let text = languages[language][i18nKey];

            if (!text) {
                console.warn(`Missing i18n key data: ${language}.${i18nKey}`);
                text = `${language}.${i18nKey}`;
            }

            e.innerText = text;
        }
    })
}

const popup = document.getElementById("popup");
const noInfoPopup = document.getElementById("no-info-popup");

function closePopup() {
    popup.hidden = true;
    noInfoPopup.hidden = true;
}

const scenarioSelect = document.getElementById("scenario-select");
const scenarioInfo = document.getElementById("scenario-info");
const scenarioName = document.getElementById("scenario-name");
const scenarioImage = document.getElementById("scenario-image");
const scenarioVideoUrl = document.getElementById("scenario-video-url");
const scenarioData = document.getElementById("scenario-data");
const scenarioInitiationPointLatitude = document.getElementById("scenario-initiation-point-latitude");
const scenarioInitiationPointLongitude = document.getElementById("scenario-initiation-point-longitude");
const scenarioNumberOfPoints = document.getElementById("scenario-number-of-points");
const scenarioRuptureVelocity = document.getElementById("scenario-rupture-velocity");
const scenarioSurfaceArea = document.getElementById("scenario-surface-area");
const scenarioMagnitude = document.getElementById("scenario-magnitude");
const scenarioSeismicMoment = document.getElementById("scenario-seismic-moment");
const scenarioFocalMechanism = document.getElementById("scenario-focal-mechanism");
const scenarioDisplacementDistribution = document.getElementById("scenario-displacement-distribution");
const scenarioTimeSymmetry = document.getElementById("scenario-time-symmetry");

function prepareScenarios() {
    for (const key in scenarios) {
        if (scenarios.hasOwnProperty(key)) {
            const option = document.createElement("option");
            option.setAttribute("value", key);
            option.innerText = scenarios[key].name;
            option.classList.add("i18n");
            option.setAttribute("data-i18n-key", `${key}-name`);
            scenarioSelect.appendChild(option);
        }
    }
}

const openInfoCard = document.getElementById("open-info-card");
const infoCard = document.getElementById("info-card");
const closeInfoCard = document.getElementById("close-info-card");

openInfoCard.onclick = () => {
    openInfoCard.hidden = true;
    infoCard.hidden = false;
};

closeInfoCard.onclick = () => {
    infoCard.hidden = true;
    openInfoCard.hidden = false;
};

const popupLatitude = document.getElementById("popup-latitude");
const popupLongitude = document.getElementById("popup-longitude");
const popupImage = document.getElementById("popup-image");
const popupData = document.getElementById("popup-data");
const popupComponent = document.getElementById("popup-component");
const popupWidth = 400;
const popupHeight = 360;
const noInfoPopupWidth = 220;
const noInfoPopupHeight = 60;
const popupDistanceFromScreenEdge = 5;
const popupDistanceFromPointX = popupWidth / 2 + 10;
const popupDistanceFromPointY = popupHeight / 2 + 10;
const noInfoPopupDistanceFromPointX = noInfoPopupWidth / 2 + 10;
const noInfoPopupDistanceFromPointY = noInfoPopupHeight / 2 + 10;

popup.style.width = popupWidth.toString();
popup.style.height = popupHeight.toString();
noInfoPopup.style.width = noInfoPopupWidth;
noInfoPopup.style.height = noInfoPopupHeight;

const allowedAreaUpperLeft = [17.5619, 42.9438];
const allowedAreaUpperRight = [18.4997, 42.9176];
const allowedAreaLowerLeft = [17.5373, 42.3346];
const allowedAreaLowerRight = [18.4677, 42.3101];

function loadOpenLayers() {
    const center = ol.proj.fromLonLat([18.015, 42.625]);
    const allowedAreaCoords = [
        ol.proj.fromLonLat(allowedAreaUpperLeft),
        ol.proj.fromLonLat(allowedAreaUpperRight),
        ol.proj.fromLonLat(allowedAreaLowerRight),
        ol.proj.fromLonLat(allowedAreaLowerLeft)
    ];
    const boundsObject = {
        type: "FeatureCollection",
        crs: {
            type: "name",
            properties: {
                name: "EPSG:4326"
            }
        },
        features: [
            {
                type: "Feature",
                geometry: {
                    type: "Polygon",
                    coordinates: [
                        [
                            ol.proj.fromLonLat([180, 90]),
                            ol.proj.fromLonLat([180, -90]),
                            ol.proj.fromLonLat([-180, -90]),
                            ol.proj.fromLonLat([-180, 90])
                        ],
                        allowedAreaCoords
                    ]
                }
            }
        ]
    };
    const backdropStyle = [
        new ol.style.Style({
            stroke: new ol.style.Stroke({
                color: "transparent",
                width: 1
            }),
            fill: new ol.style.Fill({
                color: "rgba(0, 0, 0, 0.5)"
            })
        })
    ];
    const selectedPointSource = new ol.source.Vector();
    const selectedPointStyle = [
        new ol.style.Style({
            image: new ol.style.Circle({
                stroke: new ol.style.Stroke({
                    color: "rgb(201, 25, 25)",
                    width: 2
                }),
                fill: new ol.style.Fill({
                    color: "rgb(238, 13, 13)"
                }),
                radius: 5
            })
        })
    ];

    const selectedHoverPointStyle = [
        new ol.style.Style({
            image: new ol.style.Circle({
                stroke: new ol.style.Stroke({
                    color: "rgba(201, 25, 25, 0.5)",
                    width: 2
                }),
                fill: new ol.style.Fill({
                    color: "rgba(238, 13, 13, 0.5)"
                }),
                radius: 5
            })
        })
    ];

    const draw = new ol.interaction.Draw({
        source: selectedPointSource,
        type: "Point",
        style: selectedHoverPointStyle
    });

    const map = new ol.Map({
        target: "map",
        layers: [
            new ol.layer.Tile({
                source: new ol.source.OSM(),
            }),
            new ol.layer.Vector({
                source: new ol.source.Vector({
                    features: new ol.format.GeoJSON().readFeatures(boundsObject)
                }),
                style: backdropStyle
            }),
            new ol.layer.Vector({
                source: selectedPointSource,
                style: selectedPointStyle
            })
        ],
        view: new ol.View({
            center: center,
            zoom: 10
        })
    });

    const allowedArea = new ol.format.GeoJSON().readFeatures({
        type: "FeatureCollection",
        crs: {
            type: "name",
            properties: {
                name: "EPSG:4326"
            }
        },
        features: [
            {
                type: "Feature",
                geometry: {
                    type: "Polygon",
                    coordinates: [allowedAreaCoords]
                }
            }
        ]
    })[0].getGeometry();

    function calcPopupPosition(coord, popupDimension, windowDimension) {
        if (coord + (popupDimension / 2) + popupDistanceFromScreenEdge > windowDimension) {
            return `${windowDimension - popupDimension - popupDistanceFromScreenEdge}px`;
        } else if (coord - (popupDimension / 2) - popupDistanceFromScreenEdge < 0) {
            return `${popupDistanceFromScreenEdge}px`;
        } else {
            return `${coord - popupDimension / 2}px`;
        }
    }

    function distance(lat1, lon1, lat2, lon2) {
        return Math.sqrt(Math.pow(lat1 - lat2, 2) + Math.pow(lon1 - lon2, 2));
    }

    function findClosestPoint(latitude, longitude, points) {
        const sortedPoints = points.map(p => {
            return {
                point: p,
                distance: distance(latitude, longitude, p.latitude, p.longitude)
            }
        }).sort((a, b) => a.distance - b.distance);

        return sortedPoints[0].point;
    }

    let lastFeature = null;

    selectedPointSource.on("addfeature", (e) => {
        if (!!lastFeature) {
            selectedPointSource.removeFeature(lastFeature);
        }

        lastFeature = e.feature;
    });

    scenarioSelect.onchange = () => {
        if (lastFeature === null) {
            map.addInteraction(draw);
        }

        closePopup();

        const scenario = scenarios[scenarioSelect.value];

        scenarioInfo.hidden = false;
        scenarioName.innerText = scenario.name;
        scenarioImage.setAttribute("src", scenario.image);
        scenarioImage.setAttribute("alt", scenarioSelect.value);
        scenarioVideoUrl.setAttribute("href", scenario.videoUrl);
        scenarioData.setAttribute("action", scenario.data);
        scenarioData.hidden = false;
        scenarioInitiationPointLatitude.innerText = scenario.initiationPoint.latitude;
        scenarioInitiationPointLongitude.innerText = scenario.initiationPoint.longitude;
        scenarioNumberOfPoints.innerText = scenario.numberOfPoints;
        scenarioRuptureVelocity.innerText = scenario.ruptureVelocity;
        scenarioSurfaceArea.innerText = scenario.surfaceArea;
        scenarioMagnitude.innerText = scenario.magnitude;
        scenarioSeismicMoment.innerHTML = scenario.seismicMoment.replace("⨯10", "⨯10<sup>") + "</sup>";
        scenarioFocalMechanism.innerText = scenario.focalMechanism;
        scenarioDisplacementDistribution.innerText = scenario.displacementDistribution;
        scenarioTimeSymmetry.innerText = scenario.timeSymmetry;

        translateLang(lang);
    };

    map.on("click", (e) => {
        closePopup();

        if (!!scenarioSelect.value) {
            const [lon, lat] = ol.proj.toLonLat(e.coordinate);
            const [x, y] = e.pixel;
            const scenario = scenarios[scenarioSelect.value];

            if (allowedArea.intersectsCoordinate(e.coordinate)) {
                const closestPoint = findClosestPoint(lat, lon, scenario.points);
                const [cLon, cLat] = [+closestPoint.longitude, +closestPoint.latitude];

                popupComponent.onchange = () => {
                    const imageAttr = `image${popupComponent.value}`;
                    popupImage.hidden = true;
                    popupImage.src = `${scenario.imagesDir}/${closestPoint[imageAttr]}`;
                    popupImage.alt = `${closestPoint.id}_${popupComponent.value}`;
                    popupImage.hidden = false;
                };

                popup.hidden = false;
                popupLatitude.innerText = ol.coordinate.toStringHDMS([0, lon]).replace(" 0° 00′ 00″", "");
                popupLongitude.innerText = ol.coordinate.toStringHDMS([lat, 0]).replace("0° 00′ 00″ ", "");
                popup.style.left = calcPopupPosition(x + popupDistanceFromPointX, popupWidth, window.innerWidth);
                popup.style.top = calcPopupPosition(y - popupDistanceFromPointY, popupHeight, window.innerHeight);

                const imageAttr = `image${popupComponent.value}`;

                popupImage.setAttribute("src", `${scenario.imagesDir}/${closestPoint[imageAttr]}`);
                popupImage.setAttribute("alt", `${closestPoint.id}_${popupComponent.value}`);
                popupData.setAttribute("href", `${scenario.imagesDir}/${closestPoint.data}`);
                popupData.setAttribute("download", `${closestPoint.id}.txt`);
            } else {
                noInfoPopup.hidden = false;
                noInfoPopup.style.left =
                    calcPopupPosition(x + noInfoPopupDistanceFromPointX, noInfoPopupWidth, window.innerWidth);
                noInfoPopup.style.top =
                    calcPopupPosition(y - noInfoPopupDistanceFromPointY, noInfoPopupHeight, window.innerHeight);
            }
        }
    });
}

window.onload = () => {
    prepareScenarios();
    loadOpenLayers();
}
