def predict_loan_approval(model, input_data):
    prediction = model.predict([input_data])
    return "Approved" if prediction[0] == 1 else "Rejected"
