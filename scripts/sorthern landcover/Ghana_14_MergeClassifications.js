//=======================================================================================
//STEP 1: Import Layers
//=======================================================================================
// Spatial parameters
var aoi = ee.FeatureCollection("projects/ee-boatennana200/assets/Southern_Ghana_Dissolve")
reserve = reserve.map(function(feats){return feats.difference(geometry)})
Map.addLayer(reserve, {}, "Ghana");


var composite = ee.Image('projects/ee-mayeh/assets/Ghana_Composite');
var waterClass = ee.Image('projects/landcover-mit/assets/version_2/Ghana_Water')//.select('classification').remap([1],[2]).byte();
var mangClass = ee.Image('projects/landcover-mit/assets/version_2/Ghana_Mangrove')//.select('classification').remap([1],[3]).byte();
var wetlandClass = ee.Image('projects/landcover-mit/assets/version_2/Ghana_Wetland')//.select('classification').remap([1],[4]).byte();
var miningClass = ee.Image('projects/landcover-mit/assets/version_2/Ghana_Mining')//.select('classification').remap([1],[5]).byte();
var artSurfClass = ee.Image('projects/landcover-mit/assets/version_2/Ghana_ArtificialSurfaces')//.select('classification').remap([1],[6]).byte();
var cForestClass = ee.Image('projects/landcover-mit/assets/version_2/Ghana_ClosedForest')//.select('classification').remap([1],[7]).byte();
var woodyCropsClass = ee.Image('projects/landcover-mit/assets/version_2/Ghana_WoodyCrops')//.select('classification').remap([1],[9]).byte();
var oForestClass = ee.Image('projects/landcover-mit/assets/version_2/Ghana_OpenForest')//.select('classification').remap([1],[8]).byte();
var agClass = ee.Image('projects/landcover-mit/assets/version_2/Ghana_Agriculture')//.select('classification').remap([1],[10]).byte();
var shrublandClass = ee.Image('projects/landcover-mit/assets/version_2/Ghana_Shrubland')//.select('classification').remap([1],[11]).byte();
var agsClass = ee.Image('projects/landcover-mit/assets/version_2/Ghana_AgricSouth')//.select('classification').remap([1],[12]).byte();
var shrub2Class = ee.Image('projects/landcover-mit/assets/version_2/Ghana_Shrub_2')//.select('classification').remap([1],[13]).byte();
var palmClass = ee.Image('projects/landcover-mit/assets/version_2/Ghana_Palm')
var rubberClass = ee.Image('projects/landcover-mit/assets/version_2/Ghana_Rubber')

//Reclassify all the layers to distinct figures
waterClass = waterClass.select('classification').remap([1],[2]).byte().addBands(waterClass.select('probability'));
mangClass = mangClass.select('classification').remap([1],[3]).byte().addBands(mangClass.select('probability'));
wetlandClass = wetlandClass.select('classification').remap([1],[4]).byte().addBands(wetlandClass.select('probability'));
miningClass = miningClass.select('classification').remap([1],[5]).byte().addBands(miningClass.select('probability'));
artSurfClass = artSurfClass.select('classification').remap([1],[6]).byte().addBands(artSurfClass.select('probability'));
cForestClass = cForestClass.select('classification').remap([1],[7]).byte().addBands(cForestClass.select('probability'));
woodyCropsClass = woodyCropsClass.updateMask(rubberClass.select('classification'))//Mask out rubber areas from woody crops
                                  .updateMask(palmClass.select('classification'))// Mask out palm areas from woody crops
                                  .select('classification').remap([1],[9]).byte().addBands(woodyCropsClass.select('probability'));
oForestClass = oForestClass.select('classification').remap([1],[8]).byte().addBands(oForestClass.select('probability'))
agClass = agClass.select('classification').remap([1],[9]).byte().addBands(agClass.select('probability'));
shrublandClass = shrublandClass.select('classification').remap([1],[10]).byte().addBands(shrublandClass.select('probability'));
agsClass = agsClass.select('classification').remap([1],[9]).byte().addBands(agsClass.select('probability'));
shrub2Class = shrub2Class.select('classification').remap([1],[10]).byte().addBands(shrub2Class.select('probability'));
rubberClass = rubberClass.select('classification').remap([1], [11]).byte().addBands(rubberClass.select('probability'));
palmClass = palmClass.select('classification').remap([1], [12]).byte().addBands(palmClass.select('probability'))


//=======================================================================================
//STEP 2: Make Landcover Map
//=======================================================================================

//Merge all landcover classes with quality mask
var finalImage = ee.ImageCollection([waterClass, mangClass,wetlandClass, miningClass, artSurfClass, cForestClass, 
                                    oForestClass, woodyCropsClass, agClass, shrublandClass, agsClass, shrub2Class,
                                    rubberClass, palmClass]).qualityMosaic('probability').select('remapped');
var finalImage2 = finalImage.rename('classification').addBands(finalImage.gt(1).remap([1], [3]).rename('quality').byte())

//Create a reference image to fill gaps in the classification 
var refimage = finalImage.focalMode({radius: 3, kernelType: 'circle', units: 'pixels', iterations: 5})
var refimage2 = refimage.rename('classification').addBands(refimage.gt(1).remap([1], [2]).rename('quality').byte())

//Apply the reference image to fill the gaps
var finalllyy = ee.ImageCollection([refimage2, finalImage2]).qualityMosaic('quality').select('classification').clip(aoi)

// Set land cover palette
var palette = ['#02dcff', //water 2
              '#6749a7', //mangrove/wetlands 3
              '#aa90d3', //wetlands 4
              'black', //mining 5
              'red',//Artificial Surfaces 6
              'darkgreen', //Closed Forest 7
              '#5dbb00', //Open Forest 8
              // '#e4a14f', //Woody Crops 9 
              '#f7fb15', //Agriculture 9
              // 'tan', //Grassland 11
              'tan', //Shrub 10
              '#0a11fb', //Rubber 11
              '#fba017'  //Palm 12
              ];


var vizz = {bands: ["B4","B3","B2"],
            gamma: 1.3450000000000002,
            max: 0.1726137710571289,
            min: 0.012759654846191402,
            opacity: 1}

//Visualize the final Map
Map.addLayer(finalllyy, {min:2, max:12, palette: palette}, 'Final')

// add a land cover legend ////

var classes =  [
  'Water', 
  'Mangrove', 
  'Wetland', 
  'Mining', 
  'Built-up', 
  'Closed_Forest', 
  'Open_Forest',
  // 'Woody_Crops',
  'Agriculture',
  // 'Grassland',
  'Shrub',
  'Rubber',
  'Palm'
]

// set position
var legend = ui.Panel({
  style: {
    position: 'bottom-left',
    padding: '8px 15px'
  }
});

// set title
var legendTitle = ui.Label({
  value: 'Land Cover Class',
  style: {
    fontWeight: 'bold',
    fontSize: '15px',
    margin: '0 0 4px 0',
    padding: '0'
    }
});
legend.add(legendTitle);

// set legend rows
var makeRow = function(color, name) {
      var colorBox = ui.Label({
        style: {
          backgroundColor: color,
          // padding for height/width
          padding: '8px',
          margin: '0 0 4px 0'
        }
      });
      var description = ui.Label({
        value: name,
        style: {margin: '0 0 4px 6px'}
      });

      return ui.Panel({
        widgets: [colorBox, description],
        layout: ui.Panel.Layout.Flow('horizontal')
      });
};

// set colors/names
  for (var i = 0; i < 11; i++) {
  legend.add(makeRow(palette[i], classes[i]));
  }  

Map.add(legend);
// print(finalllyy.set(ee.Dictionary.fromLists(classes, ee.List.sequence(2, 12))).set({'year': 2024}))

Export.image.toAsset({
  image: finalllyy.set(ee.Dictionary.fromLists(classes, ee.List.sequence(2, 12))).set({'year': 2024}),
  description: 'Southern_Classified',
  region: aoi,
  scale: 30,
  maxPixels: 1e11,
  assetId: 'projects/ee-cersgisrsteams/assets/LU_LC/Southern_Classified'
})
