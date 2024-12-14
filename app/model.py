#Model placeholder

class Model:
    def __init__(self, model_path):
        # Load model here
        self.model = model_path
        
        
    def predict(self, data):
        # Perform inference here
        return data