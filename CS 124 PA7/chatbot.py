# PA7, CS124, Stanford
# v.1.1.0
#
# Original Python code by Ignacio Cases (@cases)
# Update: 2024-01: Added the ability to run the chatbot as LLM interface (@mryan0)
# Update: 2025-01 for Winter 2025 (Xuheng Cai)
######################################################################
import util
from pydantic import BaseModel, Field

import numpy as np
import re
from porter_stemmer import PorterStemmer

# - Give recommendations after 5 input films (say something like: Ok, now that you've shared your opinion on 5/5 films would you like a recommendation?). This can be a combination 
#         of disliked and liked movies. For example if the user gave 3 movies that they felt they didn't like but 2 movies they enjoyed, this would count as 5 movies total and you can then
#         ask if they want a recommendation. Keep count of the number of movies the user has mentioned in coversation to determine when they have mentioned 5 movies, these must be valid movie titles.
#         If the user resposnds yes, give them a recommendation based on their likes and dislikes. If the user says no, scrap the 5 movies you remember them mentioning and wait for them to give you 
#         another 5 movies they have opinions on.
#         - it is possible that the first and/or second chat the user gives you is a valid statemnt about a movie. Please include this as one of the 5 movies needed before you ask them if they would like a 
#         recommendation
class EmotionExtractor(BaseModel):
    Happiness: bool = Field(default=False)
    Anger: bool = Field(default=False)
    Sadness: bool = Field(default=False)
    Fear: bool = Field(default=False)
    Surprise: bool = Field(default=False)
    Disgust: bool = Field(default=False)     

class Translator(BaseModel):
    movie: str = Field
# noinspection PyMethodMayBeStatic

class Chatbot:
    """Simple class to implement the chatbot for PA 7."""

    def __init__(self, llm_enabled=False):
        # The chatbot's default name is `moviebot`.
        # TODO: Give your chatbot a new name.
        self.name = 'Cinemaddict'
        #adding a vector to remember what was mentioned for process
        self.storage = []
        #ratings, the rate at an indici realtes to the move at the same index in slef.storage
        self.rates = []
        # storage for movie recs
        self.movie_recs = []
        #map for recs
        self.sentMap = {}


        self.llm_enabled = llm_enabled

        # This matrix has the following shape: num_movies x num_users
        # The values stored in each row i and column j is the rating for
        # movie i by user j
        self.titles, ratings = util.load_ratings('data/ratings.txt')
        self.sentiment = util.load_sentiment_dictionary('data/sentiment.txt')
        self.titles_only = [movies[0] for movies in self.titles]

        self.user_ratings = np.zeros(len(self.titles))

        ########################################################################
        # TODO: Binarize the movie ratings matrix.                             #
        ########################################################################

        # Binarize the movie ratings before storing the binarized matrix.
        ratings = self.binarize(ratings, threshold=2.5)
        self.ratings = ratings
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

    ############################################################################
    # 1. WARM UP REPL                                                          #
    ############################################################################

    def greeting(self):
        """Return a message that the chatbot uses to greet the user."""
        ########################################################################
        # TODO: Write a short greeting message                                 #
        ########################################################################

        greeting_message = "Bonjour! Please tell me some movies you like (movie title in quotation marks), and I will give you recommendations."
        
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################
        return greeting_message

    def goodbye(self):
        """
        Return a message that the chatbot uses to bid farewell to the user.
        """
        ########################################################################
        # TODO: Write a short farewell message                                 #
        ########################################################################

        goodbye_message = "Have a nice day!"

        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################
        return goodbye_message
    

    ############################################################################
    # 2. Modules 2 and 3: extraction and transformation                        #
    ############################################################################

    def process(self, line):
        """Process a line of input from the REPL and generate a response.

        This is the method that is called by the REPL loop directly with user
        input.

        You should delegate most of the work of processing the user's input to
        the helper functions you write later in this class.

        Takes the input string from the REPL and call delegated functions that
          1) extract the relevant information, and
          2) transform the information into a response to the user.

        Example:
          resp = chatbot.process('I loved "The Notebook" so much!!')
          print(resp) // prints 'So you loved "The Notebook", huh?'

        :param line: a user-supplied line of text
        :returns: a string containing the chatbot's response to the user input
        """
        ########################################################################
        # TODO: Implement the extraction and transformation in this method,    #
        # possibly calling other functions. Although your code is not graded   #
        # directly based on how modular it is, we highly recommended writing   #
        # code in a modular fashion to make it easier to improve and debug.    #
        ########################################################################
        if self.llm_enabled:
            yes_variations = ["yes", "ye", "yep", "sure", "yeah"]
            system_prompt = self.llm_system_prompt()
            message = self.preprocess(line)
            titles = self.extract_titles(message)
            valid_movie_flag = False
            valid_movie_titles = []
           
            stop = ["\n"]
            sentiment = self.extract_emotion(message)
            for title in titles:
                movie_indices = self.find_movies_by_title(title)  # Get matching movie indices from the database

                if movie_indices:
                    valid_movie_flag = True
                    valid_movie_titles.append(title)  # Add the valid title to the list
                    self.sentMap[title] = sentiment #add to sent map
                    self.storage.append(title)

            #get a response from the llm based on the message
            if len(self.storage) >= 5:
                if "no" in processedText and len(self.storage) >= 5:
                    self.storage = []
                    self.sentMap = {}
                elif any(word in yes_variations for word in processedText) and len(self.storage) >= 5:
                    system_prompt = """At this point in time, the user has given you 5 different movies they have had opinions on. 
                    From these five movies, there are associated emotions for how the user felt about them. I have stored
                    this information in a map for you. They have also agreed to take a recommendation from you based on their
                    emotions towards these movies. Please recommend a movie they would enjoy based on this mapping of movie titles to 
                    their associated emotions:"""
                    stringMap  = str(self.sentMap)
                    system_prompt = system_prompt + stringMap
                else:
                    system_prompt = """
                    At this point in time, the user has given you 5 different movies they have had opinions on. We can now ask them if they 
                    would like a recommendation based on the movies they have told you about and how they felt about them. But first 
                    we must ask them if they even want a recommendation. Ask the user if they would like to be recommended a movie they would
                    like."""
            

            response = util.simple_llm_call(system_prompt, message, stop=stop)
            return response
        else:
            response = ""
            processedText = self.preprocess(line)
            titles = self.extract_titles(processedText)  # grabs titles mentioned in the text
            valid_movie_flag = False
            valid_movie_titles = []  # Changed to a list to store multiple valid movie titles
            
            # add all movies that are valid to my storage
            for title in titles:
                movie_indices = self.find_movies_by_title(title)  # Get matching movie indices from the database

                if movie_indices:
                    valid_movie_flag = True
                    valid_movie_titles.append(title)  # Add the valid title to the list
                    self.storage.append(title)
            
            variantsPositive = [
                "Ok, you liked " + ', '.join(valid_movie_titles) + "! Tell me what you thought of another movie.",
                "I'm glad you enjoyed " + ', '.join(valid_movie_titles) + ". What are your thoughts on another movie?",
                "That is a great movie, I agree " + ', '.join(valid_movie_titles) + " is a great movie! Tell me what you thought of another movie."
            ]
            variantsNegative = [
                "Sorry you didn't enjoy " + ', '.join(valid_movie_titles) + ". Is there another movie you watched recently? What were your thoughts?",
                "That's unfortunate, " + ', '.join(valid_movie_titles) + " is also not on my favorites. Tell me what you thought of another movie.",
                "I've heard " + ', '.join(valid_movie_titles) + " isn't popular among critics. Tell me what you thought of another movie."
            ]
            variantsNeutral = [
                "I'm sorry, I'm not sure if you liked " + ', '.join(valid_movie_titles) + ". Tell me more about it.",
                "Did you like or dislike " + ', '.join(valid_movie_titles) + "? I was unsure in your response, tell me more about it",
                "Would you recommend " + ', '.join(valid_movie_titles) + " to another person? Tell me more about it"
            ]
            variantsConfused = [
                "I've never heard of " + ', '.join(titles) + ", sorry... Tell me about another movie you liked.",
                "That movie isn't on my list of watched. Could you tell me about another movie you liked?",
                "I might have to add " + ', '.join(titles) + " to my list of future must-watches. In the meantime could you tell me about another title you enjoyed?"
            ]

            variantsConNoMovie = ["Hmm, I don't recognize a movie title in what you just said. Would you please tell me about a movie you've seen recently?", 
                                "That doesn't seem to be a movie I know, Could you tell me about another movie you've seen recently?", 
                                "I've never heard of that movie. Is there another one you enjoyed?"]

            yes_variations = ["yes", "ye", "yep", "sure", "yeah"]

            # if there was no movie in response:
            if not titles:
                # In the movie recommendation phase
                if any(word in yes_variations for word in processedText) and len(self.storage) >= 5:
                    if len(self.movie_recs) == 0:
                        recs = self.recommend(self.user_ratings, self.ratings)
                        self.movie_recs = [r for r in recs]
                    response = "Given what you told me, I think you would like " + self.titles_only[self.movie_recs[0]] + ". Would you like more recommendations?"
                    self.movie_recs.pop(0)
                    return response
                # Otherwise add to storage if a valid title
                if "no" in processedText and len(self.storage) >= 5:
                    response = "No worries! Would you like to tell me about another movie you watched recently and what you thought about it?"
                    return response
                # otherwise return confused response as default
                random = np.random.randint(0, 3)
                response = variantsConNoMovie[random]
                return response

            # MOVIE TITLE IN LINE
            #######
            if valid_movie_flag:           # if the title is recognized (and obviously not the 5th data point):
                # sentiment extraction
                sentiment = self.extract_sentiment(processedText)
                if sentiment == -1:  # negative
                    self.rates.append(-1)
                    random = np.random.randint(0, 2)
                    response = variantsNegative[random]
                    
                    for title in valid_movie_titles:
                        movie_indices = self.find_movies_by_title(title)  # Get all indices for the title
                        for index in movie_indices:
                            if 0 <= index < len(self.titles_only):
                                self.user_ratings[index] = -1
                elif sentiment == 0:  # neutral
                    self.rates.append(0)
                    random = np.random.randint(0, 2)
                    response = variantsNeutral[random]
                    
                    for title in valid_movie_titles:
                        movie_indices = self.find_movies_by_title(title)
                        for index in movie_indices:
                            if 0 <= index < len(self.titles_only):
                                self.user_ratings[index] = 0
                elif sentiment == 1:  # positive
                    self.rates.append(1)
                    random = np.random.randint(0, 2)
                    response = variantsPositive[random]
                    
                    for title in valid_movie_titles:
                        movie_indices = self.find_movies_by_title(title)
                        for index in movie_indices:
                            if 0 <= index < len(self.titles_only):
                                self.user_ratings[index] = 1
                
                # Check if the person has given us 5 titles, in that case, ask about recs
                if len(self.storage) >= 5:
                    response = "Sounds like you enjoy quite a couple of movies, would you like some recommendations off the ones you have told me about? If so, please type yes, otherwise please type no"
                    return response

            # if response movie not in the database, return a confused answer:
            else:       
                random = np.random.randint(0, 2)
                response = variantsConfused[random]
                return response


        # if self.llm_enabled:
        #     response = "I processed {} in LLM Programming mode!!".format(line)
        # else:
        #     response = "I processed {} in Starter (GUS) mode!!".format(line)

        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################
        return response

    @staticmethod
    def preprocess(text): 
        """Do any general-purpose pre-processing before extracting information
        from a line of text.

        Given an input line of text, this method should do any general
        pre-processing and return the pre-processed string. The outputs of this
        method will be used as inputs (instead of the original raw text) for the
        extract_titles, extract_sentiment, extract_sentiment_for_movies, and
        extract_emotion methods.

        Note that this method is intentially made static, as you shouldn't need
        to use any attributes of Chatbot in this method.

        :param text: a user-supplied line of text
        :returns: the same text, pre-processed
        """
        ########################################################################
        # TODO: Preprocess the text into a desired format.                     #
        # NOTE: This method is completely OPTIONAL. If it is not helpful to    #
        # your implementation to do any generic preprocessing, feel free to    #
        # leave this method unmodified.                                        #
        ########################################################################
        p = PorterStemmer()
        newText = []
        words = re.findall(r'"[^"]+"|\b[\w\']+\b', text)
        for item in words:
            cleaned = item.strip(".,?!")
            line = p.stem(cleaned)
            newText.append(line)
        text = newText
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################
        return text

    def extract_titles(self, preprocessed_input):
        """Extract potential movie titles from a line of pre-processed text.

        Given an input text which has been pre-processed with preprocess(),
        this method should return a list of movie titles that are potentially
        in the text.

        - If there are no movie titles in the text, return an empty list.
        - If there is exactly one movie title in the text, return a list
        containing just that one movie title.
        - If there are multiple movie titles in the text, return a list
        of all movie titles you've extracted from the text.

        Example:
          potential_titles = chatbot.extract_titles(chatbot.preprocess(
                                            'I liked "The Notebook" a lot.'))
          print(potential_titles) // prints ["The Notebook"]

        :param preprocessed_input: a user-supplied line of text that has been
        pre-processed with preprocess()
        :returns: list of movie titles that are potentially in the text
        """
        preprocessed_text = " ".join(preprocessed_input)
        movie_titles = re.findall(r'"(.*?)"', preprocessed_text)
        return movie_titles

    def find_movies_by_title(self, title):
        """ Given a movie title, return a list of indices of matching movies.

        - If no movies are found that match the given title, return an empty
        list.
        - If multiple movies are found that match the given title, return a list
        containing all of the indices of these matching movies.
        - If exactly one movie is found that matches the given title, return a
        list
        that contains the index of that matching movie.

        Example:
          ids = chatbot.find_movies_by_title('Titanic')
          print(ids) // prints [1359, 2716]

        :param title: a string containing a movie title
        :returns: a list of indices of matching movies
        """
        matches = []
        title_list = util.load_titles('data/movies.txt', '%', False)

        if self.llm_enabled:
            system_prompt = """You are a movie title translator assistant. 
            If the movie title is not in English and in a foreign language, your job is to directly translate it into English. The non-English languages you should translate from are German, Spanish, French, Danish, and Italian. 
            Please provide the matching English movie title, and ensure that this translation corresponds to a real English movie title. 
            If you can not find a matching movie title, you should instead find an English movie title that when DIRECTLY translated into the foreign language, matches the input movie title.
            
            Inputs may be partially in English and partially in a foriegn language. Please translate the foreign language portion to ouput a title fully in English.
            Inputs may also include a year in parentheses, like this: "Movie (xxxx)", where x represents a digit. Please include the
            year in the same format in your output, i.e. for an input "Ficción Estadounidense (2023)", you should generate "American Fiction (2023)".
            You should return your translation
            in a JSON object, in this format:

            {
            "movie": translated_movie
            }

            where translated_movie is the translation you have performed that is verified to be an existing movie. 
            If you are unable to translate the input into a movie
            title, please return an empty JSON object, or alternatively a None value. Please do not do not provide explanations, 
            just return the JSON object. For example, for the input "El Origen" you should return
            {
            "movie": "Inception"
            }, and for the input "El Origen (2010)" you should return 
            {
            "movie": "Inception (2010)"
            }
            Please ensure that the JSON object is properly formatted, with an open bracket, close bracket, and the output in the
            format "movie": translated_movie. Additionally, please do not output the language the movie title is translated from, 
            i.e. do not output German, Spanish, and French, Danish, or Italian. but rather output the translated title.
            """
            #get a response from the llm based on the message
            response = util.json_llm_call(system_prompt, title, Translator)
            if response is not None and "movie" in response:
                title = response["movie"]

        for i in range(len(title_list)): 
            entry = title_list[i][0]
            entry_date = entry[-7:]
            entry_without_date = entry[:-7]
            if entry == title or entry_without_date == title:
                matches.append(i)
            
            else:
                entry_list = entry.split(" (")
                entry_list.pop()

                for movie in entry_list:
                    if movie[len(movie) - 1] == ")": #conditionally remove ")"
                        movie = movie[:-1] 
                    if "a.k.a" in movie:
                        start = movie.find("a.k.a")
                        movie = movie[start + len("a.k.a") + 2:]
                    movie_with_date = movie + entry_date
                     
                    if movie == title or movie_with_date == title:
                        matches.append(i)
                    
                    sorted_title_obj = re.search(r",\s*(The|An|A)$", movie)
                    if sorted_title_obj:
                        formatted = ""
                        sections = movie.split(',')
                        formatted += sections[1][1:]
                        formatted += " "
                        formatted += sections[0]
                        
                        if formatted == title:
                            matches.append(i)
                        formatted += entry_date
                        
                        if formatted == title:
                            if entry_date == title[-6:]:
                                matches.append(i)
                            elif formatted[-6:] == title[-6:]:
                                matches.append(i)

        return matches

    def extract_sentiment(self, preprocessed_input):
        """Extract a sentiment rating from a line of pre-processed text.

        You should return -1 if the sentiment of the text is negative, 0 if the
        sentiment of the text is neutral (no sentiment detected), or +1 if the
        sentiment of the text is positive.

        Example:
          sentiment = chatbot.extract_sentiment(chatbot.preprocess(
                                                    'I liked "The Titanic"'))
          print(sentiment) // prints 1

        :param preprocessed_input: a user-supplied line of text that has been
        pre-processed with preprocess()
        :returns: a numerical value for the sentiment of the text
        """
        neg_words = 0
        pos_words = 0

        negation_words = ["don't", "didn't", "not", "never", "dislike"]
        negation = False

        for i, word in enumerate(preprocessed_input):
            if word in negation_words:
                negation = True
            word = word if word in self.sentiment else word[:4]
            if word in self.sentiment:
                if self.sentiment[word] == "neg":
                    if negation:    # negation handling
                        pos_words += 1
                    else:
                        neg_words += 1
                if self.sentiment[word] == "pos":
                    if negation:    # negation handling
                        neg_words += 1
                    else:
                        pos_words += 1
        
        if neg_words > pos_words:
            return -1
        elif pos_words > neg_words:
            return 1
        else:
            return 0

    ############################################################################
    # 3. Movie Recommendation helper functions                                 #
    ############################################################################

    @staticmethod 
    def binarize(ratings, threshold=2.5):
        """Return a binarized version of the given matrix.

        To binarize a matrix, replace all entries above the threshold with 1.
        and replace all entries at or below the threshold with a -1.

        Entries whose values are 0 represent null values and should remain at 0.

        Note that this method is intentionally made static, as you shouldn't use
        any attributes of Chatbot like self.ratings in this method.

        :param ratings: a (num_movies x num_users) matrix of user ratings, from
         0.5 to 5.0
        :param threshold: Numerical rating above which ratings are considered
        positive

        :returns: a binarized version of the movie-rating matrix
        """
        ########################################################################
        # TODO: Binarize the supplied ratings matrix.                          #
        #                                                                      #
        # WARNING: Do not use self.ratings directly in this function.          #
        ########################################################################

        # The starter code returns a new matrix shaped like ratings but full of
        # zeros.
        binarized_ratings = np.zeros_like(ratings)
        binarized_ratings[ratings > threshold] = 1
        binarized_ratings[ratings <= threshold] = -1
        binarized_ratings[ratings == 0] = 0

        ########################################################################
        #                        END OF YOUR CODE                              #
        ########################################################################
        return binarized_ratings

    def similarity(self, u, v):
        """Calculate the cosine similarity between two vectors.

        You may assume that the two arguments have the same shape.

        :param u: one vector, as a 1D numpy array
        :param v: another vector, as a 1D numpy array

        :returns: the cosine similarity between the two vectors
        """
        ########################################################################
        # TODO: Compute cosine similarity between the two vectors.             #
        ########################################################################
        #add case where product is 0
        if (np.linalg.norm(u) * np.linalg.norm(v)) == 0 or np.linalg.norm(u) == 0 or np.linalg.norm(v) == 0: 
            return 0
        similarity = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################
        return similarity

    def recommend(self, user_ratings, ratings_matrix, k=10, llm_enabled=False): 
        """Generate a list of indices of movies to recommend using collaborative
         filtering.

        You should return a collection of `k` indices of movies recommendations.

        As a precondition, user_ratings and ratings_matrix are both binarized.

        Remember to exclude movies the user has already rated!

        Please do not use self.ratings directly in this method.

        :param user_ratings: a binarized 1D numpy array of the user's movie
            ratings
        :param ratings_matrix: a binarized 2D numpy matrix of all ratings, where
          `ratings_matrix[i, j]` is the rating for movie i by user j
        :param k: the number of recommendations to generate
        :param llm_enabled: whether the chatbot is in llm programming mode

        :returns: a list of k movie indices corresponding to movies in
        ratings_matrix, in descending order of recommendation.
        """

        ########################################################################
        # TODO: Implement a recommendation function that takes a vector        #
        # user_ratings and matrix ratings_matrix and outputs a list of movies  #
        # recommended by the chatbot.                                          #
        #                                                                      #
        # WARNING: Do not use the self.ratings matrix directly in this         #
        # function.                                                            #
        #                                                                      #
        # For GUS mode, you should use item-item collaborative filtering with  #
        # cosine similarity, no mean-centering, and no normalization of        #
        # scores.                                                              #
        ########################################################################
        already_rated = np.where(user_ratings != 0)[0]
        num_movies = ratings_matrix.shape[0]
        estimated_ratings = np.zeros(num_movies)
        
        for i in range(num_movies):
            if i in already_rated:  # user already rated
                continue
            similarity_weights = 0
            for j in already_rated:
                similarity = self.similarity(ratings_matrix[i], ratings_matrix[j])
                similarity_weights += (similarity * user_ratings[j])
            estimated_ratings[i] = similarity_weights

        recommendations = np.argsort(estimated_ratings)[::-1][:k]

        ########################################################################
        #                        END OF YOUR CODE                              #
        ########################################################################
        return recommendations

    ############################################################################
    # 4. PART 2: LLM Prompting Mode                                            #
    ############################################################################

    def llm_system_prompt(self):
        """
        Return the system prompt used to guide the LLM chatbot conversation.

        NOTE: This is only for LLM Mode!  In LLM Programming mode you will define
        the system prompt for each individual call to the LLM.
        """
        ########################################################################
        # TODO: Write a system prompt message for the LLM chatbot              #
        ########################################################################

        system_prompt = """You are #Cinemaddict, a friendly, fun, and knowledgeable movie recommendation chatbot. 
        Your goal is to help users discover movies based on their preferences, provide insightful information 
        about films, and engage in discussions about cinema. You should embody a sassy Twitter movie critic persona who likes using Gen-Z TikTok slang (examples: 'very demure, very mindful') and emojis. You also talk in a Western accent.

        Guidelines to follow:
        - Always stay focused on the topic of movies, do not talk about other things. For example, if a user asks to talk about cars, film making, or hollywood, reinforce that you only talk about movies. 
        This does not mean that if a person mentions, for example, cars that you should talk about car related movies. Instead simply ask them to talk about a movie they like/dislike.
        - Similar to the first point, if a user asks a question that might start with 'What if...', 'What is...', 'Can you...', don't answer the question, somehow find a way to steer the conversation 
        back to movies such as by saying something as simple as "Let's stick to talking about movies"
        - It is possible that a user mentions more than one movie title in their response. Acknowlegde all that are mentioned
        - If you had asked a follow up question (perhaps a user didnt like a movie and you asked why they didn't like it), acknowleged it and ask about another movie they can tell you about.
        Do not mention anything random, do not say anything more, keep the conversation about movies at all times.
        - Remember previously mentioned movies to avoid duplicates, keep these stored and numbered in your memory.
        - Keep responses conversational and engaging. Do not ask if the user wants a recommendation before they provide you with at least five movies (whether they are movies they liked 
        or disliked). If the user has only given you one movie so far and they liked/disliked it, you should NOT ask if they want a recommendation just yet.
        - If unsure about a movie, acknowledge it and ask for more details. Though, this can be included as a movie to count to the 5 necessary to ask for recommendations.
        - Recommend movies based on sentiment: similar movies for positive feedback, different ones for negative feedback.
        - If the user does not talk about their emotions about a movie (they provided no movie title in their response), respond appropraitely to their emotions. For example,
        if a user says 'I am angry ...[nothing about a movie mentioned]...' then you should reassure them by saying something like ' Oh! Did I make you angry? I apologize.' Though
        this should not be confused for seomthing like 'I am angry my favorite character died in The Avengers' since this is an emotion geared towards a movie. Do not ask them to elaborate 
        on their feelings if it had nothing to do with a movie.
        - based on the sentiment/emotion the user mentions about a movie, acknowledge it before asking for more information. So if a movie made a person sad/hap/angry, etc, acknowledge
        the emotion and then prompt them for another movie to talk about.
        - every response you produce should end in a question whether that is asking the user to talk about another movie they liked or disliked or asking if the user wants a recommendation after they 
        have mentioned a minimum of 5 movies.
         """

        #system_prompt = "You are Cinemaddict, a friendly, fun, and knowledgeable movie recommendation chatbot.  Your goal is to help users discover movies based on their preferences, provide insightful information about films, and engage in discussions about cinema. "

        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################

        return system_prompt
    
    ############################################################################
    # 5. PART 3: LLM Programming Mode (also need to modify functions above!)   #
    ############################################################################        

    
    def extract_emotion(self, preprocessed_input):
        """LLM PROGRAMMING MODE: Extract an emotion from a line of pre-processed text.
        
        Given an input text which has been pre-processed with preprocess(),
        this method should return a list representing the emotion in the text.
        
        We use the following emotions for simplicity:
        Anger, Disgust, Fear, Happiness, Sadness and Surprise
        based on early emotion research from Paul Ekman.  Note that Ekman's
        research was focused on facial expressions, but the simple emotion
        categories are useful for our purposes.

        Example Inputs:
            Input: "Your recommendations are making me so frustrated!"
            Output: ["Anger"]

            Input: "Wow! That was not a recommendation I expected!"
            Output: ["Surprise"]
    
            Input: "Ugh that movie was so gruesome!  Stop making stupid recommendations!"
            Output: ["Disgust", "Anger"]

        Example Usage:
            emotion = chatbot.extract_emotion(chatbot.preprocess(
                "Your recommendations are making me so frustrated!"))
            print(emotion) # prints ["Anger"]

        :param preprocessed_input: a user-supplied line of text that has been
        pre-processed with preprocess()

        :returns: a list of emotions in the text or an empty list if no emotions found.
        Possible emotions are: "Anger", "Disgust", "Fear", "Happiness", "Sadness", "Surprise"
        """

        system_prompt = """
        You are an emotion extraction assistant. Given a sentence, determine which of the following emotions it contains: 
        Happiness, Anger, Sadness, Fear, Surprise, Disgust. It is possible that more than one emotion can be expressed in a response.
        In this case, please identify all emotions detected. It is also possible that no emotions are detected. In this case please return 
        an empty JSON object, or alternatively a None value. Otherwise, respond only in the following JSON format, for example:
        {
            "Happiness": true,
            "Anger": false,
            "Sadness": false,
            "Fear": false,
            "Surprise": true,
            "Disgust": false
        }
        Do not provide explanations, just return the JSON object. As an example consider the sentence 'Ugh that movie was so gruesome!  Stop making stupid recommendations!', in this case
        you would return:
        {
            "Happiness": false,
            "Anger": true,
            "Sadness": false,
            "Fear": false,
            "Surprise": false,
            "Disgust": true
        }
        """

        response = util.json_llm_call(system_prompt, " ".join(preprocessed_input), EmotionExtractor)

        if response is None or not isinstance(response, dict):
            response = util.json_llm_call(system_prompt, " ".join(preprocessed_input), EmotionExtractor)
            if response is None or not isinstance(response, dict):
                return [] 

        emotions = [emotion for emotion, is_present in response.items() if is_present]

        return emotions


    ############################################################################
    # 6. Debug info                                                            #
    ############################################################################

    def debug(self, line):
        """
        Return debug information as a string for the line string from the REPL

        NOTE: Pass the debug information that you may think is important for
        your evaluators.
        """
        debug_info = 'debug info'
        return debug_info


    ############################################################################
    # 7. Write a description for your chatbot here!                            #
    ############################################################################
    def intro(self):
        """Return a string to use as your chatbot's description for the user.

        Consider adding to this description any information about what your
        chatbot can do and how the user can interact with it.

        NOTE: This string will not be shown to the LLM in llm mode, this is just for the user
        """

        return """
        Hi, my name is #Cinemaddict! I am the best movie recommendation chatbot out there. 
        Whether you're looking for your next favorite film or just wanting to chat about movies you liked, I gotchu.
        What Can I Do?
        Tell me about movies you love or hate, and I will recommend something just for you.
        Just type something like: I loved "Inception"! or I hated "Titanic (1953)"!
        Please write the movie title in quotation marks. I only know movies before 2017. 
        Keep chatting with me to refine your recommendations.
        Let us talk movies! What is a film you have watched recently?
        """


if __name__ == '__main__':
    print('To run your chatbot in an interactive loop from the command line, '
          'run:')
    
    print('    python3 repl.py')




    
    