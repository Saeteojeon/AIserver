
from sagemaker.model import Model

model = Model(
    image_uri='<your-ecr-image-uri>',
    role=role,
    sagemaker_session=session,
)

predictor = model.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.large',
    endpoint_name='custom-llm-endpoint'
)