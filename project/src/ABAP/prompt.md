I need to make a 


Something that takes in this table of all these different props, and then I need to do a batch API like a tutorial that I found.
Batch processing with the Batch API
The new Batch API allows to create async batch jobs for a lower price and with higher rate limits.

Batches will be completed within 24h, but may be processed sooner depending on global usage.

Ideal use cases for the Batch API include:

Tagging, captioning, or enriching content on a marketplace or blog
Categorizing and suggesting answers for support tickets
Performing sentiment analysis on large datasets of customer feedback
Generating summaries or translations for collections of documents or articles
and much more!

This cookbook will walk you through how to use the Batch API with a couple of practical examples.

We will start with an example to categorize movies using gpt-4o-mini, and then cover how we can use the vision capabilities of this model to caption images.

Please note that multiple models are available through the Batch API, and that you can use the same parameters in your Batch API calls as with the Chat Completions endpoint.

Setup
# Make sure you have the latest version of the SDK available to use the Batch API
%pip install openai --upgrade
import json
from openai import OpenAI
import pandas as pd
from IPython.display import Image, display
# Initializing OpenAI client - see https://platform.openai.com/docs/quickstart?context=python
client = OpenAI()
First example: Categorizing movies
In this example, we will use gpt-4o-mini to extract movie categories from a description of the movie. We will also extract a 1-sentence summary from this description.

We will use JSON mode to extract categories as an array of strings and the 1-sentence summary in a structured format.

For each movie, we want to get a result that looks like this:

{
    categories: ['category1', 'category2', 'category3'],
    summary: '1-sentence summary'
}
Loading data
We will use the IMDB top 1000 movies dataset for this example.

dataset_path = "data/imdb_top_1000.csv"

df = pd.read_csv(dataset_path)
df.head()
Poster_Link	Series_Title	Released_Year	Certificate	Runtime	Genre	IMDB_Rating	Overview	Meta_score	Director	Star1	Star2	Star3	Star4	No_of_Votes	Gross
0	https://m.media-amazon.com/images/M/MV5BMDFkYT...	The Shawshank Redemption	1994	A	142 min	Drama	9.3	Two imprisoned men bond over a number of years...	80.0	Frank Darabont	Tim Robbins	Morgan Freeman	Bob Gunton	William Sadler	2343110	28,341,469
1	https://m.media-amazon.com/images/M/MV5BM2MyNj...	The Godfather	1972	A	175 min	Crime, Drama	9.2	An organized crime dynasty's aging patriarch t...	100.0	Francis Ford Coppola	Marlon Brando	Al Pacino	James Caan	Diane Keaton	1620367	134,966,411
2	https://m.media-amazon.com/images/M/MV5BMTMxNT...	The Dark Knight	2008	UA	152 min	Action, Crime, Drama	9.0	When the menace known as the Joker wreaks havo...	84.0	Christopher Nolan	Christian Bale	Heath Ledger	Aaron Eckhart	Michael Caine	2303232	534,858,444
3	https://m.media-amazon.com/images/M/MV5BMWMwMG...	The Godfather: Part II	1974	A	202 min	Crime, Drama	9.0	The early life and career of Vito Corleone in ...	90.0	Francis Ford Coppola	Al Pacino	Robert De Niro	Robert Duvall	Diane Keaton	1129952	57,300,000
4	https://m.media-amazon.com/images/M/MV5BMWU4N2...	12 Angry Men	1957	U	96 min	Crime, Drama	9.0	A jury holdout attempts to prevent a miscarria...	96.0	Sidney Lumet	Henry Fonda	Lee J. Cobb	Martin Balsam	John Fiedler	689845	4,360,000
Processing step
Here, we will prepare our requests by first trying them out with the Chat Completions endpoint.

Once we're happy with the results, we can move on to creating the batch file.

categorize_system_prompt = '''
Your goal is to extract movie categories from movie descriptions, as well as a 1-sentence summary for these movies.
You will be provided with a movie description, and you will output a json object containing the following information:

{
    categories: string[] // Array of categories based on the movie description,
    summary: string // 1-sentence summary of the movie based on the movie description
}

Categories refer to the genre or type of the movie, like "action", "romance", "comedy", etc. Keep category names simple and use only lower case letters.
Movies can have several categories, but try to keep it under 3-4. Only mention the categories that are the most obvious based on the description.
'''

def get_categories(description):
    response = client.chat.completions.create(
    model="gpt-4o-mini",
    temperature=0.1,
    # This is to enable JSON mode, making sure responses are valid json objects
    response_format={ 
        "type": "json_object"
    },
    messages=[
        {
            "role": "system",
            "content": categorize_system_prompt
        },
        {
            "role": "user",
            "content": description
        }
    ],
    )

    return response.choices[0].message.content
# Testing on a few examples
for _, row in df[:5].iterrows():
    description = row['Overview']
    title = row['Series_Title']
    result = get_categories(description)
    print(f"TITLE: {title}\nOVERVIEW: {description}\n\nRESULT: {result}")
    print("\n\n----------------------------\n\n")
TITLE: The Shawshank Redemption
OVERVIEW: Two imprisoned men bond over a number of years, finding solace and eventual redemption through acts of common decency.

RESULT: {
    "categories": ["drama"],
    "summary": "Two imprisoned men develop a deep bond over the years, ultimately finding redemption through their shared acts of kindness."
}


----------------------------


TITLE: The Godfather
OVERVIEW: An organized crime dynasty's aging patriarch transfers control of his clandestine empire to his reluctant son.

RESULT: {
    "categories": ["crime", "drama"],
    "summary": "An aging crime lord hands over his empire to his hesitant son."
}


----------------------------


TITLE: The Dark Knight
OVERVIEW: When the menace known as the Joker wreaks havoc and chaos on the people of Gotham, Batman must accept one of the greatest psychological and physical tests of his ability to fight injustice.

RESULT: {
    "categories": ["action", "thriller", "superhero"],
    "summary": "Batman faces a formidable challenge as the Joker unleashes chaos on Gotham City."
}


----------------------------


TITLE: The Godfather: Part II
OVERVIEW: The early life and career of Vito Corleone in 1920s New York City is portrayed, while his son, Michael, expands and tightens his grip on the family crime syndicate.

RESULT: {
    "categories": ["crime", "drama"],
    "summary": "The film depicts the early life of Vito Corleone and the rise of his son Michael within the family crime syndicate in 1920s New York City."
}


----------------------------


TITLE: 12 Angry Men
OVERVIEW: A jury holdout attempts to prevent a miscarriage of justice by forcing his colleagues to reconsider the evidence.

RESULT: {
    "categories": ["drama", "thriller"],
    "summary": "A jury holdout fights to ensure justice is served by challenging his fellow jurors to reevaluate the evidence."
}


----------------------------


Creating the batch file
The batch file, in the jsonl format, should contain one line (json object) per request. Each request is defined as such:

{
    "custom_id": <REQUEST_ID>,
    "method": "POST",
    "url": "/v1/chat/completions",
    "body": {
        "model": <MODEL>,
        "messages": <MESSAGES>,
        // other parameters
    }
}
Note: the request ID should be unique per batch. This is what you can use to match results to the initial input files, as requests will not be returned in the same order.

# Creating an array of json tasks

tasks = []

for index, row in df.iterrows():
    
    description = row['Overview']
    
    task = {
        "custom_id": f"task-{index}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            # This is what you would have in your Chat Completions API call
            "model": "gpt-4o-mini",
            "temperature": 0.1,
            "response_format": { 
                "type": "json_object"
            },
            "messages": [
                {
                    "role": "system",
                    "content": categorize_system_prompt
                },
                {
                    "role": "user",
                    "content": description
                }
            ],
        }
    }
    
    tasks.append(task)
# Creating the file

file_name = "data/batch_tasks_movies.jsonl"

with open(file_name, 'w') as file:
    for obj in tasks:
        file.write(json.dumps(obj) + '\n')
Uploading the file
batch_file = client.files.create(
  file=open(file_name, "rb"),
  purpose="batch"
)
print(batch_file)
FileObject(id='file-lx16f1KyIxQ2UHVvkG3HLfNR', bytes=1127310, created_at=1721144107, filename='batch_tasks_movies.jsonl', object='file', purpose='batch', status='processed', status_details=None)
Creating the batch job
batch_job = client.batches.create(
  input_file_id=batch_file.id,
  endpoint="/v1/chat/completions",
  completion_window="24h"
)
Checking batch status
Note: this can take up to 24h, but it will usually be completed faster.

You can continue checking until the status is 'completed'.

batch_job = client.batches.retrieve(batch_job.id)
print(batch_job)
Retrieving results
result_file_id = batch_job.output_file_id
result = client.files.content(result_file_id).content
result_file_name = "data/batch_job_results_movies.jsonl"

with open(result_file_name, 'wb') as file:
    file.write(result)
# Loading data from saved file
results = []
with open(result_file_name, 'r') as file:
    for line in file:
        # Parsing the JSON string into a dict and appending to the list of results
        json_object = json.loads(line.strip())
        results.append(json_object)
Reading results
Reminder: the results are not in the same order as in the input file. Make sure to check the custom_id to match the results against the input requests

# Reading only the first results
for res in results[:5]:
    task_id = res['custom_id']
    # Getting index from task id
    index = task_id.split('-')[-1]
    result = res['response']['body']['choices'][0]['message']['content']
    movie = df.iloc[int(index)]
    description = movie['Overview']
    title = movie['Series_Title']
    print(f"TITLE: {title}\nOVERVIEW: {description}\n\nRESULT: {result}")
    print("\n\n----------------------------\n\n")
TITLE: American Psycho
OVERVIEW: A wealthy New York City investment banking executive, Patrick Bateman, hides his alternate psychopathic ego from his co-workers and friends as he delves deeper into his violent, hedonistic fantasies.

RESULT: {
    "categories": ["thriller", "psychological", "drama"],
    "summary": "A wealthy investment banker in New York City conceals his psychopathic alter ego while indulging in violent and hedonistic fantasies."
}


----------------------------


TITLE: Lethal Weapon
OVERVIEW: Two newly paired cops who are complete opposites must put aside their differences in order to catch a gang of drug smugglers.

RESULT: {
    "categories": ["action", "comedy", "crime"],
    "summary": "An action-packed comedy about two mismatched cops teaming up to take down a drug smuggling gang."
}


----------------------------


TITLE: A Star Is Born
OVERVIEW: A musician helps a young singer find fame as age and alcoholism send his own career into a downward spiral.

RESULT: {
    "categories": ["drama", "music"],
    "summary": "A musician's career spirals downward as he helps a young singer find fame amidst struggles with age and alcoholism."
}


----------------------------


TITLE: From Here to Eternity
OVERVIEW: In Hawaii in 1941, a private is cruelly punished for not boxing on his unit's team, while his captain's wife and second-in-command are falling in love.

RESULT: {
    "categories": ["drama", "romance", "war"],
    "summary": "A drama set in Hawaii in 1941, where a private faces punishment for not boxing on his unit's team, amidst a forbidden love affair between his captain's wife and second-in-command."
}


----------------------------


TITLE: The Jungle Book
OVERVIEW: Bagheera the Panther and Baloo the Bear have a difficult time trying to convince a boy to leave the jungle for human civilization.

RESULT: {
    "categories": ["adventure", "animation", "family"],
    "summary": "An adventure-filled animated movie about a panther and a bear trying to persuade a boy to leave the jungle for human civilization."
}


----------------------------


 This batch API takes in this information, takes in all of the columns and then will process each one by adding a specific prompt adds these prompts into the batch what it's going to do is it's going to take all that info and then it's going to process it 

instead of the imdb and movie examples it will take in data from data/product_data.csv

 and make the meta type title, the product description and the meta description. The product description needs to be formatted beautifully and use HTML to do it The meta description has to be 160 characters long. The meta title can be, you know, use your best judgment. I need to make a Pydantic model for the product description. 
 
 I'm going to use JSON structured output in the API. And please do your best 

Product description.
Meta title
Meta description