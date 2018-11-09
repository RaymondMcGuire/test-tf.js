var __extends = (this && this.__extends) || (function () {
    var extendStatics = Object.setPrototypeOf ||
        ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
        function (d, b) { for (var p in b) if (b.hasOwnProperty(p)) d[p] = b[p]; };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : new P(function (resolve) { resolve(result.value); }).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
var __generator = (this && this.__generator) || function (thisArg, body) {
    var _ = { label: 0, sent: function() { if (t[0] & 1) throw t[1]; return t[1]; }, trys: [], ops: [] }, f, y, t, g;
    return g = { next: verb(0), "throw": verb(1), "return": verb(2) }, typeof Symbol === "function" && (g[Symbol.iterator] = function() { return this; }), g;
    function verb(n) { return function (v) { return step([n, v]); }; }
    function step(op) {
        if (f) throw new TypeError("Generator is already executing.");
        while (_) try {
            if (f = 1, y && (t = y[op[0] & 2 ? "return" : op[0] ? "throw" : "next"]) && !(t = t.call(y, op[1])).done) return t;
            if (y = 0, t) op = [0, t.value];
            switch (op[0]) {
                case 0: case 1: t = op; break;
                case 4: _.label++; return { value: op[1], done: false };
                case 5: _.label++; y = op[1]; op = [0]; continue;
                case 7: op = _.ops.pop(); _.trys.pop(); continue;
                default:
                    if (!(t = _.trys, t = t.length > 0 && t[t.length - 1]) && (op[0] === 6 || op[0] === 2)) { _ = 0; continue; }
                    if (op[0] === 3 && (!t || (op[1] > t[0] && op[1] < t[3]))) { _.label = op[1]; break; }
                    if (op[0] === 6 && _.label < t[1]) { _.label = t[1]; t = op; break; }
                    if (t && _.label < t[2]) { _.label = t[2]; _.ops.push(op); break; }
                    if (t[2]) _.ops.pop();
                    _.trys.pop(); continue;
            }
            op = body.call(thisArg, _);
        } catch (e) { op = [6, e]; y = 0; } finally { f = t = 0; }
        if (op[0] & 5) throw op[1]; return { value: op[0] ? op[1] : void 0, done: true };
    }
};
/* =========================================================================
 *
 *  neural_network_model.ts
 *
 *
 *
 * ========================================================================= */
var EcognitaNeuralNetwork;
(function (EcognitaNeuralNetwork) {
    var NeuralNetworkModel = /** @class */ (function () {
        function NeuralNetworkModel() {
        }
        return NeuralNetworkModel;
    }());
    EcognitaNeuralNetwork.NeuralNetworkModel = NeuralNetworkModel;
})(EcognitaNeuralNetwork || (EcognitaNeuralNetwork = {}));
/* =========================================================================
 *
 *  style_transfer_model.ts
 *  deep
 *
 *
 * ========================================================================= */
/// <reference path="./neural_network_model.ts" />
var EcognitaNeuralNetwork;
(function (EcognitaNeuralNetwork) {
    var StyleTransferModel = /** @class */ (function (_super) {
        __extends(StyleTransferModel, _super);
        function StyleTransferModel(contentImagePath, styleImagePath, outputImagePath) {
            var _this = _super.call(this) || this;
            _this.contentImagePath = contentImagePath;
            _this.styleImagePath = styleImagePath;
            _this.outputImagePath = outputImagePath;
            _this.STYLE_LAYERS = [
                ['block1_conv1', 0.2],
                ['block2_conv1', 0.2],
                ['block3_conv1', 0.2],
                ['block4_conv1', 0.2],
                ['block5_conv1', 0.2]
            ];
            _this.MEANS = tf.tensor1d([123.68, 116.779, 103.939]).reshape([1, 1, 1, 3]);
            return _this;
        }
        StyleTransferModel.prototype.getLayerResult = function (model, input, layerName) {
            var currentResult = input;
            var idx = 1;
            while (true) {
                var layer = model.getLayer(null, idx++);
                if (layer.name === layerName)
                    return layer.apply(currentResult);
                currentResult = layer.apply(currentResult);
            }
        };
        StyleTransferModel.prototype.computeContentCost = function (rawContentActivation, generatedContentActivation) {
            return tf.tidy(function () {
                var _a = generatedContentActivation.shape, nH = _a[1], nW = _a[2], nC = _a[3];
                var rawContentActivationUnrolled = tf.transpose(rawContentActivation).reshape([nC, nH * nW]);
                var generatedContentActivationUnrolled = tf.transpose(generatedContentActivation).reshape([nC, nH * nW]);
                var contentCost = tf.mul((1 / (4 * nH * nW * nC)), tf.square(generatedContentActivationUnrolled.sub(rawContentActivationUnrolled)).sum());
                return contentCost;
            });
        };
        StyleTransferModel.prototype.computeGramMatrix = function (activation) {
            var GramMatrix = tf.matMul(activation, tf.transpose(activation));
            return GramMatrix;
        };
        StyleTransferModel.prototype.computeLayerStyleCost = function (rawContentActivation, generatedContentActivation) {
            var _this = this;
            return tf.tidy(function () {
                var _a = generatedContentActivation.shape, nH = _a[1], nW = _a[2], nC = _a[3];
                rawContentActivation = tf.transpose(rawContentActivation)
                    .reshape([nC, nH * nW]);
                generatedContentActivation = tf.transpose(generatedContentActivation)
                    .reshape([nC, nH * nW]);
                var rawContentGramMatrix = _this.computeGramMatrix(rawContentActivation);
                var generatedContentGramMatrix = _this.computeGramMatrix(generatedContentActivation);
                var layerStyleCost = tf.mul((1 / (4 * nC * nC * nH * nH * nW * nW)), tf.square(rawContentGramMatrix.sub(generatedContentGramMatrix)).sum());
                return layerStyleCost;
            });
        };
        StyleTransferModel.prototype.computeStyleCost = function (model, inputImage, generatedImage) {
            var _this = this;
            return tf.tidy(function () {
                var styleCost = tf.scalar(0);
                for (var _i = 0, _a = _this.STYLE_LAYERS; _i < _a.length; _i++) {
                    var _b = _a[_i], layerName = _b[0], coeff = _b[1];
                    var activation = _this.getLayerResult(model, inputImage, layerName);
                    var generatedActivation = _this.getLayerResult(model, generatedImage, layerName);
                    var layerCost = _this.computeLayerStyleCost(activation, generatedActivation);
                    styleCost = styleCost.add(tf.scalar(coeff).mul(layerCost));
                }
                return styleCost;
            });
        };
        StyleTransferModel.prototype.computeLossFunction = function (contentCost, styleCost, alpha, beta) {
            if (alpha === void 0) { alpha = 10; }
            if (beta === void 0) { beta = 40; }
            return tf.scalar(alpha).mul(contentCost).add(tf.scalar(beta).mul(styleCost));
        };
        StyleTransferModel.prototype.generateNoiseImage = function (image, noiseRatio) {
            if (noiseRatio === void 0) { noiseRatio = 0.6; }
            var noiseImage = tf.randomUniform([1, 100, 100, 3], -20, 20);
            return noiseImage.mul(noiseRatio).add(image.mul(1 - noiseRatio)).variable();
        };
        StyleTransferModel.prototype.loadImage = function (path) {
            return __awaiter(this, void 0, void 0, function () {
                var img, p;
                return __generator(this, function (_a) {
                    switch (_a.label) {
                        case 0: return [4 /*yield*/, Jimp.read(path)];
                        case 1:
                            img = _a.sent();
                            img.resize(100, 100);
                            p = [];
                            img.scan(0, 0, img.bitmap.width, img.bitmap.height, function (x, y, idx) {
                                p.push(this.bitmap.data[idx + 0]);
                                p.push(this.bitmap.data[idx + 1]);
                                p.push(this.bitmap.data[idx + 2]);
                            });
                            return [2 /*return*/, tf.tensor3d(p, [100, 100, 3]).reshape([1, 100, 100, 3]).sub(this.MEANS)];
                    }
                });
            });
        };
        StyleTransferModel.prototype.saveImage = function (path, tensor) {
            var newTensor = tensor.add(this.MEANS).reshape([100, 100, 3]);
            var newTensorArray = Array.from(newTensor.dataSync());
            var i = 0;
            return new Promise(function (resolve, reject) {
                // eslint-disable-next-line no-new
                new Jimp(100, 100, function (err, image) {
                    if (err)
                        return reject(err);
                    image.scan(0, 0, image.bitmap.width, image.bitmap.height, function (x, y, idx) {
                        this.bitmap.data[idx + 0] = newTensorArray[i++];
                        this.bitmap.data[idx + 1] = newTensorArray[i++];
                        this.bitmap.data[idx + 2] = newTensorArray[i++];
                        this.bitmap.data[idx + 3] = 255;
                    });
                    image.getBase64(Jimp.MIME_JPEG, function (err, src) {
                        var img = document.createElement("img");
                        img.setAttribute("src", src);
                        document.body.appendChild(img);
                    });
                    return resolve(null);
                });
            });
        };
        StyleTransferModel.prototype.run = function () {
            return __awaiter(this, void 0, void 0, function () {
                var _this = this;
                var vgg19, currentImage, styleImage, rawActivation, outputImage, loss, optimizer, i, start, cost;
                return __generator(this, function (_a) {
                    switch (_a.label) {
                        case 0: return [4 /*yield*/, tf.loadModel("./vgg19-tensorflowjs-model/model/model.json")];
                        case 1:
                            vgg19 = _a.sent();
                            return [4 /*yield*/, this.loadImage(this.contentImagePath)];
                        case 2:
                            currentImage = _a.sent();
                            return [4 /*yield*/, this.loadImage(this.styleImagePath)];
                        case 3:
                            styleImage = _a.sent();
                            rawActivation = this.getLayerResult(vgg19, currentImage, 'block4_conv2');
                            outputImage = this.generateNoiseImage(currentImage);
                            loss = function () {
                                var contentCost = _this.computeContentCost(rawActivation, _this.getLayerResult(vgg19, outputImage, 'block4_conv2'));
                                var styleCost = _this.computeStyleCost(vgg19, styleImage, outputImage);
                                var totalCost = _this.computeLossFunction(contentCost, styleCost, 10, 40);
                                return totalCost;
                            };
                            optimizer = tf.train.adam(2);
                            i = 0;
                            _a.label = 4;
                        case 4:
                            if (!(i < 1000)) return [3 /*break*/, 8];
                            start = Date.now();
                            cost = optimizer.minimize(function () { return loss(); }, true, [outputImage]);
                            if (!(i % 100 == 0)) return [3 /*break*/, 6];
                            return [4 /*yield*/, this.saveImage(this.outputImagePath, outputImage)];
                        case 5:
                            _a.sent();
                            _a.label = 6;
                        case 6:
                            console.log("epoch: " + (i + 1) + "/1000, cost: " + cost.dataSync() + ", use " + (Date.now() - start) / 1000 + "s");
                            _a.label = 7;
                        case 7:
                            i++;
                            return [3 /*break*/, 4];
                        case 8: return [2 /*return*/];
                    }
                });
            });
        };
        return StyleTransferModel;
    }(EcognitaNeuralNetwork.NeuralNetworkModel));
    EcognitaNeuralNetwork.StyleTransferModel = StyleTransferModel;
})(EcognitaNeuralNetwork || (EcognitaNeuralNetwork = {}));
/* =========================================================================
 *
 *  main.ts
 *  deep
 *
 *
 * ========================================================================= */
/// <reference path="./style_transfer_model.ts" />
var style_transfer_network = new EcognitaNeuralNetwork.StyleTransferModel("./images/louvre.jpg", "./images/Starry_Night.jpg", "./images/output.jpg");
style_transfer_network.run();
// (async function () {
//     style_transfer_network.run();
// })().catch(console.error)
