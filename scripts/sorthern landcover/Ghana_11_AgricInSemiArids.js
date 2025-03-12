//=======================================================================================
//STEP 1: Set Parameters 
//=======================================================================================

// Spatial parameters
var aoi = ee.FeatureCollection("projects/ee-boatennana200/assets/Southern_Ghana_Dissolve")
// Map.addLayer(aoi, {}, "Ghana", false);


// Load Layers 
var ag = ag20 ;
var notAg = notAg20 ;
var composite = ee.Image('projects/ee-mayeh/assets/Ghana_Composite');
var waterClass = ee.Image('projects/landcover-mit/assets/version_2/Ghana_Water').select('classification');
var mangClass = ee.Image('projects/landcover-mit/assets/version_2/Ghana_Mangrove').select('classification');
var wetlandClass = ee.Image('projects/landcover-mit/assets/version_2/Ghana_Wetland').select('classification');
var miningClass = ee.Image('projects/landcover-mit/assets/version_2/Ghana_Mining').select('classification');
var artSurfClass = ee.Image('projects/landcover-mit/assets/version_2/Ghana_ArtificialSurfaces').select('classification');
var cForestClass = ee.Image('projects/landcover-mit/assets/version_2/Ghana_ClosedForest').select('classification');
var woodyCropsClass = ee.Image('projects/landcover-mit/assets/version_2/Ghana_WoodyCrops').select('classification');
var oForestClass = ee.Image('projects/landcover-mit/assets/version_2/Ghana_OpenForest').select('classification');

Map.setOptions('satellite');

//=======================================================================================
//STEP 2: Mask Landsat Image 
//=======================================================================================

//Mask of previously classified pixels
var base = ee.Image.constant(0);
var classMask = waterClass.unmask().or(mangClass.unmask())
                .or(miningClass.unmask()).or(artSurfClass.unmask())
                .or(cForestClass.unmask()).or(woodyCropsClass.unmask())
                .or(oForestClass.unmask()).or(wetlandClass.unmask())
                .not().clip(aoi);
Map.addLayer(classMask, {palette:['white','blue']}, 'Class Mask', false);

// Mask the composite we are going to use for classification
var compositeMasked = composite.updateMask(classMask);
Map.addLayer(compositeMasked, {bands: ['B5', 'B6', 'B4'], min: 0, max: 0.25, gamma:0.7}, 'Composite for Classification', true);
// Map.addLayer(compositeMasked, {bands: ['B6', 'B5', 'B2'], min: 0, max: 0.25, gamma:0.7}, 'Composite for Agriculture', false);
// Map.addLayer(composite,  {bands: ['B4', 'B3', 'B2'], min: 0, max: 0.2, gamma:1}, 'True Color Composite', false);


// //=======================================================================================
// //STEP 3: Classify Landsat Image 
// //=======================================================================================

// Select the predictors to be used in the Random Forest Classifier
var bands = ['B4','B5','B6','B7', 'BI', 'NDVI', 'NDMoI', 'TCW', 'SR65', 'HH', 'NDFI', 'Soil', 'GV'];

// Merge the feature collections into a single FeatureCollection.
var sites = ag.merge(notAg);

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
  'blue'  // Crops
];

// Map.addLayer (classifiedImage, {min: 0, max: 1, palette: paletteMAP}, 'Classification', false);

// =======================================================================================
// STEP 4: Refine Classification
// =======================================================================================

var filterImage = classifiedImage.reduceNeighborhood({ //run classification through a neighborhood filter
  reducer: ee.Reducer.mode(), //choose most common value in neighborhood
  kernel: ee.Kernel.square(5,'pixels') //define neighborhood
});

// Map.addLayer(filterImage,{min:0,max:1,palette: paletteMAP},'Filtered Classification', false);

// Remove Noise
var finalImage = filterImage.select(['classification_mode']).eq(1).selfMask().connectedPixelCount().gte(15).rename('classification');

Map.addLayer(finalImage, {palette: '#1687a7'}, 'Final Classification', true);



//=======================================================================================
//STEP 5: Get Probability of Classification
//=======================================================================================

var trainedClassifier = ee.Classifier.smileRandomForest(100).setOutputMode('PROBABILITY').train({
  features: training,
  classProperty: 'class',
  inputProperties: bands
});

// Classify the trained image
var probImage = compositeMasked.select(bands).classify(trainedClassifier);
// Map.addLayer (probImage, {min: 0, max: 1, palette: ['red', 'white', 'green']}, 'Classification Probability', false);

// Mask Probability to Agriculture Class
var probAg = probImage.updateMask(finalImage);
Map.addLayer (probAg, {min: 0, max: 1, palette: ['red', 'white', 'green']}, 'Classification Probability', false);

// Add probability layer to export
var exportImage = finalImage.addBands(probAg.rename('probability'));
print('Final Export',exportImage);


//=======================================================================================
//STEP 6: Export Classification
//=======================================================================================


//Export the classification(s)
Export.image.toAsset({
  image: exportImage,
  description: 'Ghana_Agriculture',
  scale: 30,
  region: aoi,
  maxPixels:1e13
});
