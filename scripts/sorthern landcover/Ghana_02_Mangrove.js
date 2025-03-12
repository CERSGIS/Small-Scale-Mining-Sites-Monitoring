//=======================================================================================
//STEP 1: Set Parameters 
//=======================================================================================

// Spatial parameters
var aoi = ee.FeatureCollection("projects/ee-boatennana200/assets/Southern_Ghana_Dissolve")

// Map.addLayer(aoi, {}, "Ghana", false);

// Set Parameters - 
var mangrove = mangrove20;
var notMangrove = notMangrove20 ;
var composite = ee.Image('projects/ee-mayeh/assets/Ghana_Composite');
var waterClass = ee.Image('projects/landcover-mit/assets/version_2/Ghana_Water').select('classification');


Map.setOptions('satellite');

//=======================================================================================
//STEP 2: Mask Landsat Image 
//=======================================================================================

// View reference mangrove layer
var mangLiu = mangroveLiu.mosaic();
Map.addLayer(mangLiu, {}, "Mangrove - Liu, 2020", false);
Map.addLayer(mangGIRI, {}, "Mangrove - GIRI, 2000", false);
Map.addLayer(mangGMW2016, {}, "Mangrove - GMW, 2016", false);
Map.addLayer(mangGMW2010, {}, "Mangrove - GMW, 2010", false);

// mask to lower elevation regions
var elevationMask = ee.Image("USGS/SRTMGL1_003").select('elevation').lt(50).clip(aoi);
// Map.addLayer(elevationMask.selfMask(), {palette:'blue'}, 'Elevation Mask', false);

// mask out by NDMI
var NDMIMask = composite.select(['NDMI']).lt(0.25);
// Map.addLayer(NDMIMask.selfMask(), {palette:'green'}, 'NDMI Mask', false);

//Mask of previously classified pixels
var base = ee.Image.constant(0);
var classMask = waterClass.unmask(base).not().clip(aoi);
// Map.addLayer(classMask, {palette:['white','blue']}, 'Class Mask', false);

// Mask the composite we are going to use for classification
var compositeMasked = composite.updateMask(classMask).updateMask(elevationMask).updateMask(NDMIMask);
Map.addLayer(compositeMasked, {bands: ['B5', 'B6', 'B4'], min: 0, max: 0.25, gamma:0.7}, 'Composite for Classification');


//=======================================================================================
//STEP 3: Classify Landsat Image 
//=======================================================================================

// //Test out bands for classification
// Map.addLayer(compositeMasked, {bands: ['ndviSD'], min:-1, max:1, palette: ['red', 'orange','white', 'green', 'blue']}, "Test Index", false);
// Map.addLayer(compositeMasked, {bands: ['dryB5', 'dryB6', 'dryB4'], min: 0, max: 0.25, gamma:0.7}, 'Dry Composite', false);

// Select the predictors to be used in the Random Forest Classifier
var bands = ['B3','B4','B5','B6','B7', 'NDMI', 'NDVI', 'TCW', 'MSAVI', 'HH', 'HV'];

// Merge the feature collections into a single FeatureCollection.
var sites = mangrove.merge(notMangrove);

var training = compositeMasked.select(bands).sampleRegions({
  collection: sites,
  properties: ['class'],
  scale: 30,
});

var trainedClassifier = ee.Classifier.smileRandomForest(100).train({
  features: training,
  classProperty: 'class',
  inputProperties: bands
});

// Classify the trained image
var classifiedImage = compositeMasked.select(bands).classify(trainedClassifier);

// Create palette
var paletteMAP = [
  'black',  // Other
  'blue'  // Mangrove
];

Map.addLayer (classifiedImage, {min: 0, max: 1, palette: paletteMAP}, 'Classification', false);

//=======================================================================================
//STEP 4: Refine Classification
//=======================================================================================

var filterImage = classifiedImage.reduceNeighborhood({ //run classification through a neighborhood filter
  reducer: ee.Reducer.mode(), //choose most common value in neighborhood
  kernel: ee.Kernel.square(2,'pixels') //define neighborhood
});
// Map.addLayer(filterImage,{min:0,max:1,palette: paletteMAP},'Filtered Classification', false);
 
// Remove Noise
var finalImage = filterImage.select(['classification_mode']).eq(1).selfMask().connectedPixelCount().gte(20).rename('classification');
Map.addLayer(finalImage, {palette: '#1687a7'}, 'Final Classification');

//=======================================================================================
//STEP 6: Get Probability of Classification
//=======================================================================================

var trainedClassifier = ee.Classifier.smileRandomForest(100).setOutputMode('PROBABILITY').train({
  features: training,
  classProperty: 'class',
  inputProperties: bands
});

// Classify the trained image
var probImage = compositeMasked.select(bands).classify(trainedClassifier);
// Map.addLayer (probImage, {min: 0, max: 1, palette: ['red', 'white', 'green']}, 'Classification Probability', false);

// Mask Probability to Mangrove Class
var probMangrove = probImage.updateMask(finalImage);
Map.addLayer (probMangrove, {min: 0, max: 1, palette: ['red', 'white', 'green']}, 'Mangrove Probability', false);

// Add probability layer to export
var exportImage = finalImage.addBands(probMangrove.rename('probability'));
print('Final Export',exportImage);


//=======================================================================================
//STEP 5: Export Classification
//=======================================================================================

//Export the classification(s)
Export.image.toAsset({
  image: exportImage,
  description: 'Ghana_Mangrove',
  scale: 30,
  region: aoi,
  maxPixels:1e13
}); //pyramid policy: sample