# Insert here the initialization code as outlined on this page:
# https://docs.clarifai.com/api-guide/api-overview/api-clients#client-installation-instructions
##############################################################################################
# Installation
##############################################################################################

# python -m pip install clarifai-grpc

##############################################################################################
# Initialize the gRPC-based client to communicate with the Clarifai platform.
##############################################################################################

# Import the Clarifai gRPC-based objects needed
from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel
from clarifai_grpc.grpc.api import resources_pb2, service_pb2, service_pb2_grpc
from clarifai_grpc.grpc.api.status import status_pb2, status_code_pb2

# Construct the communications channel
channel = ClarifaiChannel.get_grpc_channel()
# Construct the V2Stub object for accessing all the Clarifai API functionality
stub = service_pb2_grpc.V2Stub(channel)

##############################################################################################
# This is where you set up the metadata object that's used to authenticate.
# This authorization will be used by every Clarifai API call.
# Change the following authorization key to your own credentials
# Example: metadata = (('authorization', 'Key ' + 'a123457612345678'),)
##############################################################################################

# metadata = (('authorization', 'Key ' + 'YOUR_CLARIFAI_PAT_HERE'),)
# Or, if you were to use an API Key:
metadata = (('authorization', 'Key ' + '8cbc4a0f585242f1843c5689261fbab4'),)
# Yes, the word 'Key' appears in addition to the alphanumeric PAT or API Key

##############################################################################################
# A UserAppIDSet object is needed when using a PAT. It contains two pieces of information:
# user_id (your user id) and app_id (app id that contains the model of interest).
# Both of them are specified as string values.
##############################################################################################

userDataObject = resources_pb2.UserAppIDSet(user_id='d1or2kkmhik1', app_id='my-first-application')

post_model_outputs_response = stub.PostModelOutputs(
    service_pb2.PostModelOutputsRequest(
        user_app_id=userDataObject,  # The userDataObject is created in the overview and is required when using a PAT
        model_id="general-english-image-caption-blip",  # This is model ID of the clarifai/main General model.
        version_id="0a62e100eb4f481ebd37cbcc16a3be0e",  # This is optional. Defaults to the latest model version.
        inputs=[
            resources_pb2.Input(
                data=resources_pb2.Data(
                    image=resources_pb2.Image(
                        url="woz1copy.jpeg"
                    )
                )
            )
        ]
    ),
    metadata=metadata
)

if post_model_outputs_response.status.code != status_code_pb2.SUCCESS:
    print("There was an error with your request!")
    print("\tCode: {}".format(post_model_outputs_response.outputs[0].status.code))
    print("\tDescription: {}".format(post_model_outputs_response.outputs[0].status.description))
    print("\tDetails: {}".format(respopost_model_outputs_responsense.outputs[0].status.details))
    raise Exception("Post model outputs failed, status: " + post_model_outputs_response.status.description)

# Since we have one input, one output will exist here.
output = post_model_outputs_response.outputs[0]

print("Predicted concepts:")
for concept in output.data.concepts:
    print("\t%s %.2f" % (concept.name, concept.value))
