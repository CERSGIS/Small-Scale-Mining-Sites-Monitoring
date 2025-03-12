//=======================================================================================
//STEP 1: Set Parameters 
//=======================================================================================

// Spatial parameters
var shrubland = ee.FeatureCollection("projects/ee-boatennana200/assets/Southern_Ghana_Dissolve")
// Map.addLayer(shrubland, {}, "Ghana", false);

// Load Layers 
var bare = bare20 ;
var veg = veg20 ;
var composite = ee.Image('projects/ee-mayeh/assets/Ghana_Composite');
var waterClass = ee.Image('projects/landcover-mit/assets/version_2/Ghana_Water').select('classification');
var mangClass = ee.Image('projects/landcover-mit/assets/version_2/Ghana_Mangrove').select('classification');
var wetlandClass = ee.Image('projects/landcover-mit/assets/version_2/Ghana_Wetland').select('classification');
var miningClass = ee.Image('projects/landcover-mit/assets/version_2/Ghana_Mining').select('classification');
var artSurfClass = ee.Image('projects/landcover-mit/assets/version_2/Ghana_ArtificialSurfaces').select('classification');
var cForestClass = ee.Image('projects/landcover-mit/assets/version_2/Ghana_ClosedForest').select('classification');
var woodyCropsClass = ee.Image('projects/landcover-mit/assets/version_2/Ghana_WoodyCrops').select('classification');
var oForestClass = ee.Image('projects/landcover-mit/assets/version_2/Ghana_OpenForest').select('classification');
var agClass = ee.Image('projects/landcover-mit/assets/version_2/Ghana_Agriculture').select('classification');
var shrubland = ee.Image('projects/landcover-mit/assets/version_2/Ghana_Shrubland').select('classification');

Map.setOptions('satellite');

//=======================================================================================
//STEP 2: Mask Landsat Image 
//=======================================================================================

//Mask of previously classified pixels
var base = ee.Image.constant(0);
var classMask = waterClass.unmask(base).or(mangClass.unmask())
                .or(miningClass.unmask()).or(artSurfClass.unmask())
                .or(cForestClass.unmask()).or(woodyCropsClass.unmask())
                .or(oForestClass.unmask()).or(agClass.unmask())
                .or(shrubland.unmask()).or(wetlandClass.unmask())
                .not().clip(shrubland);
Map.addLayer(classMask, {palette:['white','blue']}, 'Class Mask', false);

// Mask the composite we are going to use for classification
var compositeMasked = composite.updateMask(classMask).updateMask(ee.Image().paint(table).remap([0], [1]).unmask(0).not().selfMask());
Map.addLayer(composite,  {bands: ['B4', 'B3', 'B2'], min: 0, max: 0.2, gamma:1}, 'True Color Composite', false);
Map.addLayer(compositeMasked, {bands: ['B5', 'B6', 'B4'], min: 0, max: 0.25, gamma:0.7}, 'Composite for Classification', true);


//=======================================================================================
//STEP 3: Classify Landsat Image 
//=======================================================================================

// Select the predictors to be used in the Random Forest Classifier
var bands = ['B4','B5','B6','B7', 'BI', 'NDVI', 'NDMoI', 'MSAVI', 'SR65', 'EVI', 'HH'];

// Merge the feature collections into a single FeatureCollection.
var sites = veg.merge(bare);

var training = compositeMasked.select(bands).sampleRegions({
  collection: sites,
  properties: ['class'],
  scale: 30,
});

var trainedClassifier = ee.Classifier.smileRandomForest({
    numberOfTrees: 150,
    minLeafPopulation: 1,
    bagFraction: 0.5}).train({
  features: training,
  classProperty: 'class',
  inputProperties: bands
});

// Classify the trained image
var classifiedImage = compositeMasked.select(bands).classify(trainedClassifier);

// Create palette
var paletteMAP = [
  'green',  // Vegetation
  'blue'  // Bare
];

Map.addLayer (classifiedImage, {min: 0, max: 1, palette: paletteMAP}, 'Classification', false);

// Clean up classification
var filterImage = classifiedImage.reduceNeighborhood({ //run classification through a neighborhood filter
  reducer: ee.Reducer.mode(), //choose most common value in neighborhood
  kernel: ee.Kernel.square(2,'pixels') //define neighborhood
});

Map.addLayer(filterImage,{min:0,max:1,palette: paletteMAP},'Filtered Classification', false);

// // =======================================================================================
// // STEP 4: Split Classification into Vegetation and Bare
// // =======================================================================================

var remap = filterImage.remap([0,1],[1,2]);
// print(remap)

var finalBare = remap.eq(1).selfMask().rename('classification');
Map.addLayer(finalBare.selfMask(), {palette: 'tan'}, "Bare");

var finalVeg = remap.eq(2).selfMask().rename('classification');
Map.addLayer(finalVeg.selfMask(), {palette: 'lightgreen'}, "Vegetation");


//=======================================================================================
//STEP 5: Get Probability of Classification
//=======================================================================================

// Train the classifier for probability - Vegetation
var trainedClassifier = ee.Classifier.smileRandomForest({
    numberOfTrees: 150,
    minLeafPopulation: 1,
    bagFraction: 0.5}).setOutputMode('PROBABILITY').train({
  features: training,
  classProperty: 'class',
  inputProperties: bands
});

// Classify the trained image
var probImage = compositeMasked.select(bands).classify(trainedClassifier);

// Mask Probability to Vegetation Class
var probVeg = probImage.updateMask(finalVeg);
Map.addLayer (probVeg, {min: 0, max: 1, palette: ['red', 'white', 'green']}, 'Classification Probability - Vegetation', false);

// Add probability layer to export
var exportImageVeg = finalVeg.addBands(probVeg.rename('probability'));
// print('Final Export',exportImage);

var training = compositeMasked.select(bands).sampleRegions({
  collection: sites,
  properties: ['class2'],
  scale: 30,
});

// Train the classifier for probability - Bare
var trainedClassifier = ee.Classifier.smileRandomForest(100).setOutputMode('PROBABILITY').train({
  features: training,
  classProperty: 'class2',
  inputProperties: bands
});

// Classify the trained image
var probImage = compositeMasked.select(bands).classify(trainedClassifier);

// Mask Probability to Bare Class
var probBare = probImage.updateMask(finalBare);
Map.addLayer (probBare, {min: 0, max: 1, palette: ['red', 'white', 'green']}, 'Classification Probability - Bare', false);

// Add probability layer to export
var exportImageBare = finalBare.addBands(probBare.rename('probability'));
// print('Final Export',exportImage);
// 
// //=======================================================================================
//STEP 6: Export Classification
//=======================================================================================


//Export the classification(s)
Export.image.toAsset({
  image: exportImageVeg,
  description: 'Ghana_AgricSouth',
  scale: 30,
  region: shrubland,
  maxPixels:1e13
});

//Export the classification(s)
Export.image.toAsset({
  image: exportImageBare,
  description: 'Ghana_Shrub_2',
  scale: 30,
  region: shrubland,
  maxPixels:1e13
});

