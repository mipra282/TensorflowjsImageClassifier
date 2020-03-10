let model;

// if there is a change to "Upload Image" button, 
// load and render the image
$("#select-file-image").change(function () {
    renderImage(this.files[0]);
});

// renders the image which is loaded from disk to the img tag 
function renderImage(file) {
    var reader = new FileReader();
    reader.onload = function (event) {
        img_url = event.target.result;
        document.getElementById("test-image").src = img_url;
    }
    reader.readAsDataURL(file);
}


async function loadModel() {
    console.log("model loading..");


    loader = document.getElementById("progress-box");
    load_button = document.getElementById("load-button");
    loader.style.display = "block";


    modelName = "mobilenet";


    model = undefined;


    model = await tf.loadLayersModel('output/model.json');


    loader.style.display = "none";
    load_button.disabled = true;
    load_button.innerHTML = "Loaded Model";
    console.log("model loaded..");
}

// preprocess the image to be mobilenet friendly
function preprocessImage(image, modelName) {

    // resize the input image to mobilenet's target size of (224, 224)
    let tensor = tf.browser.fromPixels(image)
        .resizeNearestNeighbor([224, 224])
        .toFloat();

    // if model is not available, send the tensor with expanded dimensions
    if (modelName === undefined) {
        return tensor.expandDims();
    }

    // if model is mobilenet, feature scale tensor image to range [-1, 1]
    else if (modelName === "mobilenet") {
        let offset = tf.scalar(127.5);
        return tensor.sub(offset)
            .div(offset)
            .expandDims();
    }

    // else throw an error
    else {
        alert("Unknown model name..")
    }
}

// If "Predict Button" is clicked, preprocess the image and
// make predictions using mobilenet
$("#predict-button").click(async function () {
    // check if model loaded
    if (model == undefined) {
        alert("Please load the model first..")
    }

    // check if image loaded
    if (document.getElementById("predict-box").style.display == "none") {
        alert("Please load an image using 'Demo Image' or 'Upload Image' button..")
    }

    // html-image element can be given to tf.fromPixels
    let image = document.getElementById("test-image");
    let tensor = preprocessImage(image, modelName);

    // make predictions on the preprocessed image tensor
    let predictions = await model.predict(tensor).data();

    // get the model's prediction results
    let results = Array.from(predictions)
        .map(function (p, i) {
            return {
                probability: p,
                className: IMAGENET_CLASSES[i]
            };
        }).sort(function (a, b) {
            return b.probability - a.probability;
        }).slice(0, 5);

    // display the top-1 prediction of the model
    document.getElementById("results-box").style.display = "block";
    document.getElementById("prediction").innerHTML = "MobileNet prediction - <b>" + results[0].className + "</b>";

    // display top-5 predictions of the model
    var ul = document.getElementById("predict-list");
    ul.innerHTML = "";
    results.forEach(function (p) {
        console.log(p.className + " " + p.probability.toFixed(6));
        var li = document.createElement("LI");
        li.innerHTML = p.className + " " + p.probability.toFixed(6);
        ul.appendChild(li);
    });
});