/* =========================================================================
 *
 *  main.ts
 *  deep 
 *  
 *  
 * ========================================================================= */
/// <reference path="./style_transfer_model.ts" />
let style_transfer_network =new EcognitaNeuralNetwork.StyleTransferModel("./images/louvre.jpg","./images/Starry_Night.jpg","./images/output.jpg");
style_transfer_network.run();
// (async function () {
//     style_transfer_network.run();
// })().catch(console.error)