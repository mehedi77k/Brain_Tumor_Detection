const uploadBox = document.getElementById("uploadBox");
const imageInput = document.getElementById("imageInput");
const previewImage = document.getElementById("previewImage");
const uploadPlaceholder = document.getElementById("uploadPlaceholder");
const analyzeBtn = document.getElementById("analyzeBtn");

const emptyState = document.getElementById("emptyState");
const loadingState = document.getElementById("loadingState");
const errorState = document.getElementById("errorState");
const errorMessage = document.getElementById("errorMessage");
const resultContent = document.getElementById("resultContent");

const predictionBadge = document.getElementById("predictionBadge");
const predictionName = document.getElementById("predictionName");
const confidenceValue = document.getElementById("confidenceValue");
const confidenceStatus = document.getElementById("confidenceStatus");

const tumorGrade = document.getElementById("tumorGrade");
const tumorDescription = document.getElementById("tumorDescription");
const recommendationText = document.getElementById("recommendationText");

const gradcamSection = document.getElementById("gradcamSection");
const gradcamImage = document.getElementById("gradcamImage");

let selectedFile = null;
let probabilityChart = null;


// =====================================================
// Upload Handling
// =====================================================

uploadBox.addEventListener("click", () => {
    imageInput.click();
});

imageInput.addEventListener("change", (event) => {
    const file = event.target.files[0];

    if (!file) {
        return;
    }

    selectedFile = file;

    const imageUrl = URL.createObjectURL(file);

    previewImage.src = imageUrl;
    previewImage.classList.remove("hidden");
    uploadPlaceholder.classList.add("hidden");

    analyzeBtn.disabled = false;

    resetResultPanel();
});


// =====================================================
// Analyze Button
// =====================================================

analyzeBtn.addEventListener("click", async () => {
    if (!selectedFile) {
        showError("Please upload an MRI image first.");
        return;
    }

    const formData = new FormData();
    formData.append("image", selectedFile);

    showLoading();

    try {
        const response = await fetch("/predict", {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        if (!response.ok || !data.success) {
            throw new Error(data.error || "Prediction failed.");
        }

        showResult(data);

    } catch (error) {
        showError(error.message);
    }
});


// =====================================================
// UI State Functions
// =====================================================

function resetResultPanel() {
    emptyState.classList.remove("hidden");
    loadingState.classList.add("hidden");
    errorState.classList.add("hidden");
    resultContent.classList.add("hidden");
}

function showLoading() {
    emptyState.classList.add("hidden");
    errorState.classList.add("hidden");
    resultContent.classList.add("hidden");
    loadingState.classList.remove("hidden");
}

function showError(message) {
    emptyState.classList.add("hidden");
    loadingState.classList.add("hidden");
    resultContent.classList.add("hidden");

    errorMessage.textContent = message;
    errorState.classList.remove("hidden");
}

function showResult(data) {
    emptyState.classList.add("hidden");
    loadingState.classList.add("hidden");
    errorState.classList.add("hidden");
    resultContent.classList.remove("hidden");

    predictionBadge.textContent = data.predicted_class.toUpperCase();
    predictionName.textContent = data.display_name;
    confidenceValue.textContent = `${data.confidence}%`;

    if (data.confidence_status) {
        confidenceStatus.innerHTML = `
            <strong>${data.confidence_status.level}</strong><br/>
            ${data.confidence_status.message}
        `;
    }

    tumorGrade.textContent = data.grade || "";
    tumorDescription.textContent = data.description || "";
    recommendationText.textContent = data.recommendation || "";

    renderProbabilityChart(data.probabilities);

    if (data.gradcam) {
        gradcamImage.src = data.gradcam;
        gradcamSection.classList.remove("hidden");
    } else {
        gradcamSection.classList.add("hidden");
    }
}


// =====================================================
// Chart Rendering
// =====================================================

function renderProbabilityChart(probabilities) {
    const ctx = document.getElementById("probabilityChart").getContext("2d");

    const labels = Object.keys(probabilities).map(formatClassName);
    const values = Object.values(probabilities);

    if (probabilityChart) {
        probabilityChart.destroy();
    }

    probabilityChart = new Chart(ctx, {
        type: "bar",
        data: {
            labels: labels,
            datasets: [
                {
                    label: "Probability (%)",
                    data: values,
                    backgroundColor: [
                        "rgba(24, 168, 79, 0.85)",
                        "rgba(52, 199, 113, 0.82)",
                        "rgba(123, 211, 152, 0.82)",
                        "rgba(170, 232, 194, 0.88)"
                    ],
                    borderColor: [
                        "rgba(8, 114, 54, 1)",
                        "rgba(27, 148, 83, 1)",
                        "rgba(65, 161, 99, 1)",
                        "rgba(100, 170, 125, 1)"
                    ],
                    borderWidth: 1.5,
                    borderRadius: 10,
                    barThickness: 54
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,

            plugins: {
                legend: {
                    display: false
                },

                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return `${context.raw}%`;
                        }
                    }
                }
            },

            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,

                    ticks: {
                        color: "#486957",
                        stepSize: 10
                    },

                    grid: {
                        color: "rgba(31, 157, 85, 0.10)"
                    }
                },

                x: {
                    ticks: {
                        color: "#355645",
                        font: {
                            weight: "700"
                        }
                    },

                    grid: {
                        display: false
                    }
                }
            }
        }
    });
}


// =====================================================
// Helper
// =====================================================

function formatClassName(name) {
    if (name === "notumor") {
        return "No Tumor";
    }

    return name.charAt(0).toUpperCase() + name.slice(1);
}