background1='''Is the lighting in the image natural (e.g., sunlight) or artificial (e.g., indoor lighting)?
Does the image appear to be taken indoors or outdoors?
Are there any notable shadows cast in the image?
Does the image appear to be taken during the day or at night? '''

context1='''What is the predominant color of the object? What is the shape of the object? What is the size of the object? What is the texture of the object's surface? '''

background2='''What is the environment or background of the object in the image? Does the image appear to be taken indoors or outdoors? What are the surrounding objects or elements in the image? Does the image appear to be taken during the day or at night? '''

genreal1 = '''Look at the background area around the object in the image, and then use this information and your prior knowledge to infer the following : 
'''

genreal2 = '''Look at the object in the picture and its surrounding background area, and then based on the information about the object itself and other background information to answer the following question: '''

general3 = '''To determine the object in the image, focus on the central or prominent area of the image. Pay attention to the following visual features:
Shape: Look at the overall shape of the object in the image. Is it a recognizable shape, such as a square, circle, or triangle?
Color: Observe the dominant colors of the object. Note if there are any distinctive or unusual colors.
Texture: Examine the texture of the object. Is it smooth, rough, patterned, or textured in a unique way?
Size: Consider the size of the object relative to other objects in the image. Is it larger or smaller than other elements?
Context: Take into account the surroundings or background of the object. Does it provide any clues about the object's identity?
By focusing on these visual features, you can better identify and describe the object in the image.\n'''

general4 = '''You are an AI assistant who has rich visual knowledge and strong reasoning abilities.
Your goal is to help me answer a question about an image.
The question is:
Question: What is the object in the image?
You should tell me which part of the image should I focus and which visual features should I based on to answer the question. You should answer briefly.'''

general5 = "Choose the most likely answer from the given choices to answer the following questions: what is the object in the image?"

general6 = 'Choose the most likely answer from the given choices to answer the following questions. Your final answer should be in the form \boxed{{answer}}\nQuestion : what is the object in the image?'

general7 = 'Your goal is to choose the most likely answer from the given choices to answer the following questions. You should take into account the surroundings or background of the object to see whether it provides any clues about the object. Your final answer should be in the form \boxed{{answer}}\nQuestion : what is the object in the image?'

general8 = '''Is the object in the image a {}? Answer yes or no, followed by a score in 0~1 measuring the confidence of your answer.'''

general9 = '''How likely is it that the object in the picture is a {}? Answer a probability between 0 and 1.'''

general10 = '''Do you think the object in the picture is a {}? Answer a probability between 0 and 1.'''

general11