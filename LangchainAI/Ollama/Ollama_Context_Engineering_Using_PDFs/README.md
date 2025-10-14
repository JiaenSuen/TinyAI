## **Augmented QA-Bot with PDFs Source Context Engineering**

Large Language Models have powerful abilities and enormous knowledge capacity to generate text respones and answer questions, it's very useful tool for searching and organizing knowledge nowdays.

However, hallucinations in LLMs is very tricky and common problems. This issue can be similar to the "Mandela Effect" — the model may generate confident but incorrect answers because it has learned from vast data but pays less focus to rarely mentioned or unseen facts.

To address this problem, one effective method called Retrieval-Augmented Generation (RAG) was proposed in 2020.RAG retrieves relevant information from external data sources such as PDF files, websites, or databases, and integrates these retrieved contents into the prompt context. This allows the model to ground its answers on real-world data rather than relying solely on internal knowledge.

In thise tiny project, I used langchain, chroma, embedding model and llama3 . It organized a little pipline to load externel PDF files into vector databae . Then make retriever to search similar part pages form database and convert those data into context prompt. After added the prompt and user's questions together, so you can get information integration and response which more authentic and reliable .


## Framework

LangChain is a development framework for building and interacting with Large Language Models. Providing chained tasks, prompt management, data retrieval, and other features. Enable language models to understand and process information in specific fields more accurately.

Chroma is a vector database for storing and retrieving high-dimensional vector data, such as text, images, or other feature vectors.

## Model

Ollama is a tool and platform for local large-scale language model operations, providing a simple interface to call models, such as the LLaMA series

LLaMA 3 is Meta's third-generation Large Language Model (LLM), offering powerful natural language understanding and generation capabilities.

## Result Example

Question 1: What are the main foods that red foxes eat according to the fact sheet?<br>
Answer: According to the fact sheet, red foxes eat a wide variety of foods including:

* Small mammals such as voles, mice, lemmings, squirrels, hares, and rabbits
* Lake trout (up to 3 kg)
* Seal pups
* Beaver
* Reptiles
* Fruits
* Garbage
* Insects (in the summer)

Their diet also changes with the seasons:

* In fall and winter, they mainly eat small mammals.
* In spring, they supplement their diet with nesting waterfowl on prairies.
* In summer, they eat insects and berries.
--------------------------------------------------
Question 2: What are the different color variations of red foxes mentioned in the document?<br>
Answer: According to the document, the different color variations of red foxes mentioned are:

* Red (with a faint darker red line running along the back and forming a cross from shoulder to shoulder on the saddle)
* Brown
* Black
* Crossed foxes (red foxes that are browner and darker than most with a prominent dark cross on the saddle)
* Silver foxes (basically black with white-tipped guard hairs in varied amounts)
--------------------------------------------------
Question 3: How far can male red foxes travel during dispersal?<br>
Answer: According to the text, young males have been traced as far as 250 km from their birth sites.

--------------------------------------------------
Question 4: What types of sounds do red foxes make to communicate?<br>
Answer: According to the provided context, red foxes have a sharp bark, which they use when startled and to warn other foxes.

--------------------------------------------------
Question 5: What is the typical litter size of red foxes?<br>
Answer: According to the text, the typical litter size of red foxes ranges from one to 10 pups, but the average is five.

--------------------------------------------------
Question 6: How do red foxes store food for later use?<br>
Answer: According to the text, red foxes will frequently bury or hide surplus food for later use, but other animals often find and use it first.

--------------------------------------------------
Question 7: What is the conservation status of red foxes in Canada?<br>
Answer: I don't know. The provided text does not mention the conservation status of red foxes in Canada. It only discusses their management, behavior, and range.

--------------------------------------------------
Question 8: What role do red foxes play in controlling rabies?<br>
Answer: According to the provided context, red foxes can sometimes become a serious menace to public health during epidemics of rabies when they carry the disease. In such cases, attempts are made to control their populations by dropping baits containing vaccine near den sites, as seen in Ontario where some advances have been made in immunizing wild fox populations against rabies.

--------------------------------------------------
Question 9: What is a fun fact about red fox behavior mentioned in the fact sheet?<br>
Answer: According to the fact sheet, one fun fact about red fox behavior is that they will frequently bury or hide surplus food for later use, but other animals often find and use it first.

--------------------------------------------------
Question 10: Does the fact sheet mention the exact weight of an adult red fox?<br>
Answer: Yes, according to the fact sheet, adult red foxes weigh between 3.6 and 6.8 kg.