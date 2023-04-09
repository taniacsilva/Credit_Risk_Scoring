import bentoml
import numpy as np

from bentoml.io import JSON
from bentoml.io import NumpyNdarray
from pydantic import BaseModel


class CreditApplication(BaseModel):
   seniority: int
   home: str
   time: int
   age: int
   marital: str
   records: str
   job: str
   expenses: int
   income: float
   assets: float
   debt: float
   amount: int

# Pull the model as model reference (it pulls all the associate metadata of the model)
model_ref = bentoml.xgboost.get('credit_risk_model:latest')

# Call DictVectorizer object using model reference
dv = model_ref.custom_objects['DictVectorizer']

# Create the model runner (it can also scale the model separately)
model_runner = model_ref.to_runner()

# Create the service 'credit_risk_classifier' and pass the model
svc = bentoml.Service('credit_risk_classifier', runners=[model_runner])


# Define an endpoint on the BentoML service
@svc.api(input=NumpyNdarray(shape=(-1,29), dtype=np.float32, enforce_dtype= True, enforce_shape=True), output=JSON()) # decorate endpoint as in json format for input and output
async def classify(vector):
   #application_data = credit_application.dict()
   # transform data from client using dictvectorizer
   #vector = dv.transform(application_data)
   # make predictions using 'runner.predict.run(input)' instead of 'model.predict'
   prediction = await model_runner.predict.async_run(vector)
                     
   result = prediction[0] # extract prediction from 1D array
   print('Prediction:', result)
   if result > 0.5:
      return {'Status': 'DECLINED'}
   elif result > 0.3:
      return {'Status': 'MAYBE'}
   else:
      return {'Status': 'APPROVED'}
