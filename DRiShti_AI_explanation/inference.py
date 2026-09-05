
import os
import json
import numpy as np
import tensorflow as tf
import cv2


class DrishtiAI:
    """
    DrishtiAI
    Trust-Aware Explainable AI for
    Diabetic Retinopathy Screening.

    Prototype / MVP implementation.
    Not intended to replace clinical diagnosis.
    """

    def __init__(self, package_dir=None):

        if package_dir is None:
            package_dir = os.path.dirname(
                os.path.abspath(__file__)
            )

        self.package_dir = package_dir

        # Load metadata
        metadata_path = os.path.join(
            package_dir,
            "metadata.json"
        )

        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)

        # Load class names
        class_path = os.path.join(
            package_dir,
            "class_names.json"
        )

        with open(class_path, "r") as f:
            self.class_names = json.load(f)

        # Model configuration
        self.img_size = self.metadata["input_size"][0]

        # Load trained model
        model_path = os.path.join(
            package_dir,
            "drishti_ai_efficientnet.keras"
        )

        self.model = tf.keras.models.load_model(
            model_path
        )


    # ============================================================
    # IMAGE PREPROCESSING
    # ============================================================

    def preprocess_image(self, image_path):

        img = tf.keras.preprocessing.image.load_img(
            image_path,
            target_size=(
                self.img_size,
                self.img_size
            )
        )

        img_array = (
            tf.keras.preprocessing.image.img_to_array(
                img
            )
        )

        img_array = np.expand_dims(
            img_array,
            axis=0
        )

        return img_array


    # ============================================================
    # UNCERTAINTY ANALYSIS
    # ============================================================

    def calculate_uncertainty(self, probabilities):

        probabilities = np.array(probabilities)

        confidence = float(
            np.max(probabilities)
        )

        sorted_probs = np.sort(probabilities)

        prediction_margin = float(
            sorted_probs[-1] - sorted_probs[-2]
        )

        epsilon = 1e-10

        entropy = -np.sum(
            probabilities *
            np.log(probabilities + epsilon)
        )

        max_entropy = np.log(
            len(probabilities)
        )

        normalized_entropy = float(
            entropy / max_entropy
        )

        return {
            "confidence": confidence,
            "margin": prediction_margin,
            "entropy": normalized_entropy
        }


    # ============================================================
    # MODEL TRUST SCORE
    # ============================================================

    def calculate_model_trust(self, uncertainty):

        confidence = uncertainty["confidence"]
        margin = uncertainty["margin"]
        entropy = uncertainty["entropy"]

        # Higher certainty = lower entropy
        prediction_certainty = 1 - entropy

        trust_score = (
            0.50 * confidence +
            0.25 * margin +
            0.25 * prediction_certainty
        )

        trust_percentage = (
            trust_score * 100
        )

        if trust_percentage >= 75:
            trust_level = "HIGH TRUST"

        elif trust_percentage >= 50:
            trust_level = "MODERATE TRUST"

        else:
            trust_level = "LOW TRUST"

        return {
            "score": float(trust_score),
            "percentage": float(trust_percentage),
            "status": trust_level,
            "prediction_certainty": float(
                prediction_certainty
            )
        }


    # ============================================================
    # IMAGE QUALITY ASSESSMENT
    # ============================================================

    def assess_image_quality(self, img_array):

        image = img_array[0]

        # Convert RGB image to grayscale
        gray = cv2.cvtColor(
            image.astype(np.uint8),
            cv2.COLOR_RGB2GRAY
        )

        brightness = float(
            np.mean(gray)
        )

        contrast = float(
            np.std(gray)
        )

        sharpness = float(
            cv2.Laplacian(
                gray,
                cv2.CV_64F
            ).var()
        )

        # --------------------------------------------
        # Brightness score
        # Ideal brightness approximately 120
        # --------------------------------------------

        brightness_score = max(
            0,
            100 -
            abs(brightness - 120) / 120 * 100
        )

        # --------------------------------------------
        # Contrast score
        # --------------------------------------------

        contrast_score = min(
            100,
            (contrast / 60) * 100
        )

        # --------------------------------------------
        # Sharpness score
        # --------------------------------------------

        sharpness_score = min(
            100,
            (sharpness / 500) * 100
        )

        # --------------------------------------------
        # Final quality score
        # --------------------------------------------

        quality_score = (
            0.30 * brightness_score +
            0.30 * contrast_score +
            0.40 * sharpness_score
        )

        if quality_score >= 75:
            quality_level = "GOOD"

        elif quality_score >= 50:
            quality_level = "ACCEPTABLE"

        else:
            quality_level = "POOR"

        return {
            "score": float(quality_score / 100),
            "percentage": float(quality_score),
            "status": quality_level,
            "brightness": brightness,
            "contrast": contrast,
            "sharpness": sharpness,
            "brightness_score": float(
                brightness_score
            ),
            "contrast_score": float(
                contrast_score
            ),
            "sharpness_score": float(
                sharpness_score
            )
        }


    # ============================================================
    # FINAL TRUST ASSESSMENT
    # ============================================================

    def calculate_final_trust(
        self,
        model_trust,
        image_quality
    ):

        model_trust_score = (
            model_trust["score"]
        )

        image_quality_score = (
            image_quality["score"]
        )

        final_score = (
            0.70 * model_trust_score +
            0.30 * image_quality_score
        )

        final_percentage = (
            final_score * 100
        )

        if final_percentage >= 75:

            trust_level = "HIGH TRUST"

            recommendation = (
                "AI prediction can be used "
                "as a screening aid."
            )

            review_required = False

        elif final_percentage >= 50:

            trust_level = "MODERATE TRUST"

            recommendation = (
                "AI prediction should be "
                "interpreted with caution and "
                "clinical review is recommended."
            )

            review_required = True

        else:

            trust_level = "LOW TRUST"

            recommendation = (
                "Prediction should not be relied upon. "
                "Manual specialist review required."
            )

            review_required = True

        return {
            "score": float(final_score),
            "percentage": float(final_percentage),
            "status": trust_level,
            "recommendation": recommendation,
            "specialist_review_required": review_required
        }


    # ============================================================
    # REFERRAL PRIORITY ENGINE
    # ============================================================

    def determine_referral_priority(
        self,
        predicted_class,
        final_trust,
        image_quality
    ):

        trust_level = final_trust["status"]
        quality_level = image_quality["status"]

        # LOW TRUST OVERRIDE
        if trust_level == "LOW TRUST":

            return {
                "priority": (
                    "SPECIALIST REVIEW REQUIRED"
                ),
                "urgency_level": "UNCERTAIN",
                "reason": (
                    "AI prediction has low reliability. "
                    "Manual specialist review is required "
                    "before clinical referral prioritization."
                ),
                "action": (
                    "Do not rely solely on the AI prediction. "
                    "Refer the case for ophthalmologist review."
                )
            }

        # POOR IMAGE QUALITY OVERRIDE
        if quality_level == "POOR":

            return {
                "priority": (
                    "RETAKE IMAGE / REVIEW"
                ),
                "urgency_level": (
                    "IMAGE QUALITY ISSUE"
                ),
                "reason": (
                    "Image quality is insufficient for "
                    "reliable AI-assisted screening."
                ),
                "action": (
                    "Capture a better retinal image. "
                    "If image cannot be retaken, "
                    "seek specialist review."
                )
            }

        referral_map = {

            "No DR": {
                "priority": "ROUTINE SCREENING",
                "urgency_level": "LOW",
                "reason": (
                    "No signs of diabetic retinopathy "
                    "were detected by the AI screening model."
                ),
                "action": (
                    "Continue regular diabetic eye screening "
                    "according to local clinical practice."
                )
            },

            "Mild": {
                "priority": "ROUTINE REFERRAL",
                "urgency_level": "LOW-MODERATE",
                "reason": (
                    "Mild diabetic retinopathy indicators "
                    "were detected."
                ),
                "action": (
                    "Schedule an ophthalmology consultation "
                    "for further evaluation."
                )
            },

            "Moderate": {
                "priority": "PRIORITY REFERRAL",
                "urgency_level": "MODERATE",
                "reason": (
                    "Moderate diabetic retinopathy indicators "
                    "were detected."
                ),
                "action": (
                    "Arrange ophthalmology evaluation within "
                    "an appropriate priority timeframe."
                )
            },

            "Severe": {
                "priority": "URGENT REFERRAL",
                "urgency_level": "HIGH",
                "reason": (
                    "Severe diabetic retinopathy indicators "
                    "were detected."
                ),
                "action": (
                    "Prioritize ophthalmologist evaluation "
                    "as soon as possible."
                )
            },

            "Proliferative DR": {
                "priority": "URGENT REFERRAL",
                "urgency_level": "VERY HIGH",
                "reason": (
                    "Proliferative diabetic retinopathy "
                    "indicators were detected."
                ),
                "action": (
                    "Arrange urgent ophthalmologist evaluation."
                )
            }
        }

        return referral_map.get(
            predicted_class,
            {
                "priority":
                    "SPECIALIST REVIEW REQUIRED",
                "urgency_level": "UNKNOWN",
                "reason":
                    "Unable to determine referral priority.",
                "action":
                    "Seek specialist review."
            }
        )


    # ============================================================
    # GRAD-CAM
    # ============================================================



    def generate_explanation(
        self,
        prediction,
        uncertainty,
        model_trust,
        image_quality,
        final_trust,
        referral
    ):
        """
        Generate simple and technical explanations
        based on existing AI pipeline outputs.

        This method does not generate new medical predictions.
        It explains why the existing AI result and recommendation
        were produced.
        """

        # ====================================================
        # EXTRACT VALUES
        # ====================================================

        predicted_class = prediction["class_name"]

        confidence = prediction["confidence"] * 100
        margin = uncertainty["margin"] * 100
        entropy = uncertainty["entropy"]

        model_trust_percentage = model_trust["percentage"]
        model_trust_status = model_trust["status"]

        image_quality_percentage = image_quality["percentage"]
        image_quality_status = image_quality["status"]

        final_trust_percentage = final_trust["percentage"]
        final_trust_status = final_trust["status"]

        specialist_review_required = (
            final_trust["specialist_review_required"]
        )

        referral_priority = referral["priority"]
        urgency_level = referral["urgency_level"]


        # ====================================================
        # SIMPLE EXPLANATION
        # ====================================================

        if confidence >= 75:
            confidence_simple = (
                "The AI was highly confident about its prediction."
            )
        elif confidence >= 50:
            confidence_simple = (
                "The AI had moderate confidence in its prediction."
            )
        else:
            confidence_simple = (
                "The AI had low confidence in its prediction."
            )


        if entropy <= 0.35:
            uncertainty_simple = (
                "The system showed low uncertainty between possible "
                "disease categories."
            )
        elif entropy <= 0.65:
            uncertainty_simple = (
                "The system showed some uncertainty between possible "
                "disease categories."
            )
        else:
            uncertainty_simple = (
                "The system showed high uncertainty between possible "
                "disease categories."
            )


        if image_quality_status == "GOOD":
            quality_simple = (
                "The retinal image quality was good and suitable for analysis."
            )
        elif image_quality_status == "ACCEPTABLE":
            quality_simple = (
                "The retinal image quality was acceptable for analysis."
            )
        else:
            quality_simple = (
                "The retinal image quality was poor, which may reduce "
                "the reliability of the result."
            )


        if final_trust_status == "HIGH TRUST":
            trust_simple = (
                "Overall, the result is considered reliable for "
                "AI-assisted screening."
            )
        elif final_trust_status == "MODERATE TRUST":
            trust_simple = (
                "Overall, the result should be interpreted with caution."
            )
        else:
            trust_simple = (
                "Overall, the AI result is not reliable enough "
                "to be used alone."
            )


        simple_summary = (
            f"The AI screening result suggests {predicted_class}. "
            f"{confidence_simple} "
            f"{uncertainty_simple} "
            f"{quality_simple} "
            f"{trust_simple}"
        )


        # ====================================================
        # CONTEXT-AWARE RECOMMENDATION EXPLANATION
        # ====================================================

        if specialist_review_required:

            if image_quality_status == "POOR":

                recommendation_explanation = (
                    "Specialist review is recommended because the retinal "
                    "image quality may reduce the reliability of the AI result."
                )

            elif confidence < 50 and entropy > 0.65:

                recommendation_explanation = (
                    "Specialist review is recommended because the AI showed "
                    "low confidence and high uncertainty, making this result "
                    "unreliable for independent screening decisions."
                )

            else:

                recommendation_explanation = (
                    "Specialist review is recommended because the system "
                    "identified limitations in the reliability of this result."
                )

        else:

            if predicted_class == "No DR":

                recommendation_explanation = (
                    "The AI result can be used as a screening aid. Continue "
                    "regular diabetic eye screening according to local "
                    "clinical practice, while remembering that AI screening "
                    "does not replace professional medical diagnosis."
                )

            else:

                recommendation_explanation = (
                    "The AI screening result suggests a diabetic retinopathy "
                    "severity level that may require further ophthalmology "
                    "evaluation. This recommendation is based on the AI "
                    "screening result and should be confirmed through "
                    "professional clinical assessment."
                )


        # ====================================================
        # TECHNICAL EXPLANATION
        # ====================================================

        if confidence >= 75:
            confidence_interpretation = "High prediction confidence"
        elif confidence >= 50:
            confidence_interpretation = "Moderate prediction confidence"
        else:
            confidence_interpretation = "Low prediction confidence"


        if margin >= 50:
            margin_interpretation = (
                "Strong separation between the top predicted classes"
            )
        elif margin >= 20:
            margin_interpretation = (
                "Moderate separation between the top predicted classes"
            )
        else:
            margin_interpretation = (
                "Small separation between the top predicted classes"
            )


        if entropy <= 0.35:
            entropy_interpretation = "Low prediction uncertainty"
        elif entropy <= 0.65:
            entropy_interpretation = "Moderate prediction uncertainty"
        else:
            entropy_interpretation = "High prediction uncertainty"


        # ====================================================
        # DECISION BASIS
        # ====================================================

        decision_factors = []

        if confidence >= 75:
            decision_factors.append("high confidence")
        elif confidence < 50:
            decision_factors.append("low confidence")

        if entropy <= 0.35:
            decision_factors.append("low uncertainty")
        elif entropy > 0.65:
            decision_factors.append("high uncertainty")

        if image_quality_status == "GOOD":
            decision_factors.append("good image quality")
        elif image_quality_status == "POOR":
            decision_factors.append("poor image quality")

        decision_basis = " + ".join(decision_factors)

        if not decision_basis:
            decision_basis = (
                "moderate confidence and intermediate reliability factors"
            )


        # ====================================================
        # RETURN EXPLANATION
        # ====================================================

        return {

            "simple": {

                "summary": simple_summary,

                "confidence_reason": confidence_simple,

                "uncertainty_reason": uncertainty_simple,

                "image_quality_reason": quality_simple,

                "trust_reason": trust_simple,

                "recommendation_explanation":
                    recommendation_explanation
            },

            "technical": {

                "prediction": {
                    "class": predicted_class,
                    "confidence_percentage": round(confidence, 2),
                    "interpretation": confidence_interpretation
                },

                "uncertainty": {
                    "prediction_margin_percentage": round(margin, 2),
                    "normalized_entropy": round(entropy, 3),
                    "margin_interpretation": margin_interpretation,
                    "entropy_interpretation": entropy_interpretation
                },

                "image_quality": {
                    "score_percentage": round(
                        image_quality_percentage, 2
                    ),
                    "status": image_quality_status
                },

                "model_trust": {
                    "score_percentage": round(
                        model_trust_percentage, 2
                    ),
                    "status": model_trust_status
                },

                "final_trust": {
                    "score_percentage": round(
                        final_trust_percentage, 2
                    ),
                    "status": final_trust_status
                },

                "referral": {
                    "priority": referral_priority,
                    "urgency_level": urgency_level
                },

                "decision_basis": decision_basis
            }
        }


    def generate_gradcam(
        self,
        img_array,
        predicted_index
    ):

        """
        Generate Grad-CAM heatmap for the nested EfficientNetB0 model.
        """

        # Get nested EfficientNet base model
        base_model = self.model.get_layer("efficientnetb0")

        # Create model for EfficientNet internal feature maps
        last_conv_layer = base_model.get_layer("top_conv")

        feature_model = tf.keras.models.Model(
            inputs=base_model.input,
            outputs=last_conv_layer.output
        )

        # Get layers after EfficientNet in the main classifier
        gap_layer = self.model.get_layer(
            "global_average_pooling2d"
        )

        dropout_layer = self.model.get_layer(
            "dropout"
        )

        classifier_layer = self.model.get_layer(
            "dr_prediction"
        )

        # Convert image to tensor
        img_tensor = tf.convert_to_tensor(
            img_array,
            dtype=tf.float32
        )

        with tf.GradientTape() as tape:

            # Forward through EfficientNet feature extractor
            conv_outputs = feature_model(
                img_tensor,
                training=False
            )

            tape.watch(conv_outputs)

            # Forward feature maps through classifier head
            x = gap_layer(
                conv_outputs,
                training=False
            )

            x = dropout_layer(
                x,
                training=False
            )

            predictions = classifier_layer(
                x,
                training=False
            )

            class_channel = predictions[
                :,
                predicted_index
            ]

        # Calculate gradients
        grads = tape.gradient(
            class_channel,
            conv_outputs
        )

        # Global average pooling of gradients
        pooled_grads = tf.reduce_mean(
            grads,
            axis=(0, 1, 2)
        )

        # Remove batch dimension
        conv_outputs = conv_outputs[0]

        # Weighted feature maps
        heatmap = tf.reduce_sum(
            conv_outputs * pooled_grads,
            axis=-1
        )

        # ReLU
        heatmap = tf.maximum(
            heatmap,
            0
        )

        # Normalize
        max_value = tf.reduce_max(
            heatmap
        )

        if max_value > 0:
            heatmap = heatmap / max_value

        return heatmap.numpy()


    # ============================================================
    # COMPLETE ANALYSIS PIPELINE
    # ============================================================

    def analyze(
        self,
        image_path,
        generate_gradcam=True
    ):

        # 1. Preprocess
        img_array = self.preprocess_image(
            image_path
        )

        # 2. Prediction
        probabilities = self.model.predict(
            img_array,
            verbose=0
        )[0]

        predicted_index = int(
            np.argmax(probabilities)
        )

        predicted_class = (
            self.class_names[predicted_index]
        )

        confidence = float(
            np.max(probabilities)
        )

        # 3. Uncertainty
        uncertainty = (
            self.calculate_uncertainty(
                probabilities
            )
        )

        # 4. Model Trust
        model_trust = (
            self.calculate_model_trust(
                uncertainty
            )
        )

        # 5. Image Quality
        image_quality = (
            self.assess_image_quality(
                img_array
            )
        )

        # 6. Final Trust
        final_trust = (
            self.calculate_final_trust(
                model_trust,
                image_quality
            )
        )

        # 7. Referral Priority
        referral = (
            self.determine_referral_priority(
                predicted_class,
                final_trust,
                image_quality
            )
        )

        # 8. Explanation Engine
        explanation = self.generate_explanation(
            prediction={
                "class_name": predicted_class,
                "confidence": confidence
            },
            uncertainty=uncertainty,
            model_trust=model_trust,
            image_quality=image_quality,
            final_trust=final_trust,
            referral=referral
        )

        # 9. Grad-CAM
        heatmap = None

        if generate_gradcam:

            heatmap = self.generate_gradcam(
                img_array,
                predicted_index
            )

        # 9. Return complete result
        return {

            "prediction": {
                "class_index": predicted_index,
                "class_name": predicted_class,
                "confidence": confidence,
                "all_probabilities": {
                    self.class_names[i]:
                    float(probabilities[i])
                    for i in range(
                        len(self.class_names)
                    )
                }
            },

            "uncertainty": uncertainty,

            "model_trust": model_trust,

            "image_quality": image_quality,

            "final_trust": final_trust,

            "referral": referral,

            "explanation": explanation,

            "gradcam_heatmap": heatmap
        }
