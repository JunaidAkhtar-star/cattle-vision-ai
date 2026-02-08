document.addEventListener("DOMContentLoaded", function () {
  // These global variables come from Jinja in result.html
  const imageId = window.FEEDBACK_IMAGE_ID;
  const predictedBreed = window.FEEDBACK_PREDICTED_BREED;
  const predictedDisease = window.FEEDBACK_PREDICTED_DISEASE;
  const predictionConfidence = window.FEEDBACK_CONFIDENCE;
  const modelVersion = window.FEEDBACK_MODEL_VERSION;

  const btnCorrect = document.getElementById("btn-correct");
  const btnWrong = document.getElementById("btn-wrong");
  const correctionForm = document.getElementById("correction-form");
  const btnSubmitCorrection = document.getElementById("btn-submit-correction");
  const feedbackMessage = document.getElementById("feedback-message");
  const feedbackButtons = document.getElementById("feedback-buttons");

  if (!btnCorrect || !btnWrong) {
    return;
  }

  async function sendFeedback(payload) {
    try {
      const res = await fetch("/api/feedback", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });

      if (res.ok) {
        if (feedbackMessage) feedbackMessage.classList.remove("hidden");
        if (feedbackButtons) feedbackButtons.classList.add("hidden");
        if (correctionForm) correctionForm.classList.add("hidden");
      } else {
        console.error("Feedback request failed:", res.status);
      }
    } catch (err) {
      console.error("Error sending feedback:", err);
    }
  }

  btnCorrect.addEventListener("click", () => {
    sendFeedback({
      image_id: imageId,
      predicted_breed: predictedBreed,
      predicted_disease: predictedDisease,
      correct_breed: null,
      correct_disease: null,
      comment: null,
      prediction_confidence: predictionConfidence,
      model_version: modelVersion,
      is_correct: true,
    });
  });

  btnWrong.addEventListener("click", () => {
    if (correctionForm) {
      correctionForm.classList.remove("hidden");
    }
  });

  if (btnSubmitCorrection) {
    btnSubmitCorrection.addEventListener("click", () => {
      const correctBreedInput = document.getElementById("correct-breed");
      const correctDiseaseInput = document.getElementById("correct-disease");
      const commentInput = document.getElementById("feedback-comment");

      const correctBreed = correctBreedInput ? correctBreedInput.value : "";
      const correctDisease = correctDiseaseInput ? correctDiseaseInput.value : "";
      const comment = commentInput ? commentInput.value : "";

      if (!correctBreed) {
        alert("Please select the correct breed.");
        return;
      }

      sendFeedback({
        image_id: imageId,
        predicted_breed: predictedBreed,
        predicted_disease: predictedDisease,
        correct_breed: correctBreed,
        correct_disease: correctDisease || null,
        comment: comment || null,
        prediction_confidence: predictionConfidence,
        model_version: modelVersion,
        is_correct: false,
      });
    });
  }
});
