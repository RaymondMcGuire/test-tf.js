/* =========================================================================
 *
 *  style_transfer_model.ts
 *  deep 
 *  
 *  
 * ========================================================================= */
/// <reference path="./neural_network_model.ts" />
module EcognitaNeuralNetwork {
    declare var tf: any;
    declare var Jimp: any;
    export class StyleTransferModel extends NeuralNetworkModel {

        contentImagePath: string;
        styleImagePath: string;
        outputImagePath: string;

        readonly STYLE_LAYERS: any;
        readonly MEANS: any;
        constructor(contentImagePath, styleImagePath, outputImagePath) {
            super();
            this.contentImagePath = contentImagePath;
            this.styleImagePath = styleImagePath;
            this.outputImagePath = outputImagePath;

            this.STYLE_LAYERS = [
                ['block1_conv1', 0.2],
                ['block2_conv1', 0.2],
                ['block3_conv1', 0.2],
                ['block4_conv1', 0.2],
                ['block5_conv1', 0.2]
            ];

            this.MEANS = tf.tensor1d([123.68, 116.779, 103.939]).reshape([1, 1, 1, 3])
        }

        getLayerResult(model, input, layerName) {
            let currentResult = input
            let idx = 1

            while (true) {
                const layer = model.getLayer(null, idx++)
                if (layer.name === layerName) return layer.apply(currentResult)
                currentResult = layer.apply(currentResult)
            }
        }

        computeContentCost(rawContentActivation, generatedContentActivation) {
            return tf.tidy(() => {
                const [, nH, nW, nC] = generatedContentActivation.shape

                const rawContentActivationUnrolled = tf.transpose(rawContentActivation).reshape([nC, nH * nW])
                const generatedContentActivationUnrolled = tf.transpose(generatedContentActivation).reshape([nC, nH * nW])

                const contentCost = tf.mul(
                    (1 / (4 * nH * nW * nC)),
                    tf.square(
                        generatedContentActivationUnrolled.sub(rawContentActivationUnrolled)
                    ).sum()
                )

                return contentCost;
            });
        }

        computeGramMatrix(activation) {
            const GramMatrix = tf.matMul(activation, tf.transpose(activation))

            return GramMatrix
        }

        computeLayerStyleCost(rawContentActivation, generatedContentActivation) {
            return tf.tidy(() => {
                const [, nH, nW, nC] = generatedContentActivation.shape

                rawContentActivation = tf.transpose(rawContentActivation)
                    .reshape([nC, nH * nW])
                generatedContentActivation = tf.transpose(generatedContentActivation)
                    .reshape([nC, nH * nW])

                const rawContentGramMatrix = this.computeGramMatrix(rawContentActivation)
                const generatedContentGramMatrix = this.computeGramMatrix(generatedContentActivation)

                const layerStyleCost = tf.mul(
                    (1 / (4 * nC * nC * nH * nH * nW * nW)),
                    tf.square(
                        rawContentGramMatrix.sub(generatedContentGramMatrix)
                    ).sum()
                )

                return layerStyleCost;
            });
        }

        computeStyleCost(model, inputImage, generatedImage) {
            return tf.tidy(() => {
                let styleCost = tf.scalar(0)

                for (const [layerName, coeff] of this.STYLE_LAYERS) {
                    const activation = this.getLayerResult(model, inputImage, layerName)
                    const generatedActivation = this.getLayerResult(model, generatedImage, layerName)

                    const layerCost = this.computeLayerStyleCost(activation, generatedActivation)

                    styleCost = styleCost.add(tf.scalar(coeff).mul(layerCost))
                }

                return styleCost;
            });
        }

        computeLossFunction(contentCost: any, styleCost: any, alpha: number = 10, beta: number = 40) {
            return tf.scalar(alpha).mul(contentCost).add(tf.scalar(beta).mul(styleCost));
        }

        generateNoiseImage(image, noiseRatio = 0.6) {
            const noiseImage = tf.randomUniform([1, 100, 100, 3], -20, 20)

            return noiseImage.mul(noiseRatio).add(image.mul(1 - noiseRatio)).variable()
        }

        async loadImage(path) {
            let img = await Jimp.read(path)
            img.resize(100, 100)

            const p = []

            img.scan(0, 0, img.bitmap.width, img.bitmap.height, function (x, y, idx) {
                p.push(this.bitmap.data[idx + 0])
                p.push(this.bitmap.data[idx + 1])
                p.push(this.bitmap.data[idx + 2])
            })

            return tf.tensor3d(p, [100, 100, 3]).reshape([1, 100, 100, 3]).sub(this.MEANS)
        }


        saveImage(path, tensor) {
            let newTensor = tensor.add(this.MEANS).reshape([100, 100, 3])
            const newTensorArray = Array.from(newTensor.dataSync())
            let i = 0

            return new Promise(function (resolve, reject) {
                // eslint-disable-next-line no-new
                new Jimp(100, 100, function (err, image) {
                    if (err) return reject(err)
                    image.scan(0, 0, image.bitmap.width, image.bitmap.height, function (x, y, idx) {
                        this.bitmap.data[idx + 0] = newTensorArray[i++]
                        this.bitmap.data[idx + 1] = newTensorArray[i++]
                        this.bitmap.data[idx + 2] = newTensorArray[i++]
                        this.bitmap.data[idx + 3] = 255
                    })

                    image.getBase64(Jimp.MIME_JPEG, function (err, src) {
                        var img = document.createElement("img");
                        img.setAttribute("src", src);
                        document.body.appendChild(img);
                    });
                    return resolve(null)
                })
            })
        }

        async run() {

            const vgg19 = await tf.loadModel(`./vgg19-tensorflowjs-model/model/model.json`);

            const currentImage = await this.loadImage(this.contentImagePath)
            const styleImage = await this.loadImage(this.styleImagePath)

            const rawActivation = this.getLayerResult(vgg19, currentImage, 'block4_conv2')
            let outputImage = this.generateNoiseImage(currentImage)

            const loss = () => {
                const contentCost = this.computeContentCost(
                    rawActivation,
                    this.getLayerResult(vgg19, outputImage, 'block4_conv2')
                )

                const styleCost = this.computeStyleCost(vgg19, styleImage, outputImage)
                const totalCost = this.computeLossFunction(contentCost, styleCost, 10, 40)

                return totalCost;
            };

            const optimizer = tf.train.adam(2)

            for (let i = 0; i < 1000; i++) {
                const start = Date.now()
                const cost = optimizer.minimize(() => loss(), true, [outputImage])
                if(i%100==0){
                    await this.saveImage(this.outputImagePath, outputImage);
                }
                console.log(`epoch: ${i + 1}/1000, cost: ${cost.dataSync()}, use ${(Date.now() - start) / 1000}s`)
            }


        }
    }
}