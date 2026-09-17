let model = null;

const imageInput = document.getElementById("imageInput");
const previewImage = document.getElementById("previewImage");
const predictButton = document.getElementById("predictButton");
const resultContent = document.getElementById("resultContent");
const modelStatus = document.getElementById("modelStatus");

const classNames = [
    "ABBOTTS BABBLER",
    "ABBOTTS BOOBY",
    "ABYSSINIAN GROUND HORNBILL",
    "AFRICAN CROWNED CRANE",
    "AFRICAN EMERALD CUCKOO",
    "AFRICAN FIREFINCH",
    "AFRICAN OYSTER CATCHER",
    "AFRICAN PIED HORNBILL",
    "AFRICAN PYGMY GOOSE",
    "ALBATROSS",
    "ALBERTS TOWHEE",
    "ALEXANDRINE PARAKEET",
    "ALPINE CHOUGH",
    "ALTAMIRA YELLOWTHROAT",
    "AMERICAN AVOCET",
    "AMERICAN BITTERN",
    "AMERICAN COOT",
    "AMERICAN FLAMINGO",
    "AMERICAN GOLDFINCH",
    "AMERICAN KESTREL"
];

async function loadModel() {
    try {

        modelStatus.innerText = "Loading Model...";

        model = await tf.loadGraphModel(
            "./model/tfjs/model.json"
        );

        modelStatus.innerText = "Model Ready";

        console.log("Model loaded");

    } catch (error) {

        modelStatus.innerText = "Model Error";

        console.error(error);
    }
}

console.log("Image input connected:", imageInput);


// IMAGE UPLOAD
imageInput.addEventListener("change", function(event) {

    const file = event.target.files[0];

    if (!file) return;

    const reader = new FileReader();

    reader.onload = function(e) {

        previewImage.src = e.target.result;

        previewImage.style.display = "block";

        predictButton.disabled = false;
    };

    reader.readAsDataURL(file);
});


// PREDICTION
predictButton.addEventListener("click", async function() {

    if (!model) return;

    modelStatus.innerText = "Predicting...";

    const imageTensor = tf.browser
        .fromPixels(previewImage)
        .resizeBilinear([224, 224])
        .toFloat()
        .expandDims();

    const prediction = model.predict(imageTensor);

    const data = await prediction.data();

    const maxValue = Math.max(...data);

    const index = data.indexOf(maxValue);

    const confidence = (maxValue * 100).toFixed(2);


    // UNKNOWN IMAGE CHECK
    if (maxValue < 0.70) {

        resultContent.innerHTML = `
            <div class="prediction-result">

                <div class="bird-name">
                    UNKNOWN / NOT RECOGNIZED
                </div>

                <div class="confidence">
                    Confidence: ${confidence}%
                </div>

                <p>
                    This image does not confidently match
                    any of the 20 trained bird classes.
                </p>

            </div>
        `;

        modelStatus.innerText = "Unknown Image";

        tf.dispose([imageTensor, prediction]);

        return;
    }


    // KNOWN BIRD
    resultContent.innerHTML = `
        <div class="prediction-result">

            <div class="bird-name">
                ${classNames[index]}
            </div>

            <div class="confidence">
                Confidence: ${confidence}%
            </div>

            <div class="confidence-bar">

                <div
                    class="confidence-fill"
                    style="width:${confidence}%">
                </div>

            </div>

        </div>
    `;

    modelStatus.innerText = "Prediction Complete";

    tf.dispose([imageTensor, prediction]);

});


// START MODEL
loadModel();
