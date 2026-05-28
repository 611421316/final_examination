### **Columns Introduction \- Available Ingredients** 

### ---

**1\. data/user\_subset.json (Active User Subset \- Yelp Only)**

This file compiles the characteristics of the most active Yelp users (redundant source and type fields have been removed):

* **user\_id**: The unique identifier for the user (Primary Key), used to map this individual across other tables.  
* **name**: The user’s name or display nickname.  
* **yelping\_since**: The timestamp when the user registered as a Yelp member (Format: YYYY-MM-DD HH:MM:SS).  
* **review\_count**: The total number of reviews the user has published on the platform. Compared to average users, those selected for this subset typically have a very high review count.  
* **average\_stars**: The average rating given by the user across all businesses ($1.0$ to $5.0$). This can be used to determine if a user is a "harsh critic" or a "generous guest."  
* **fans**: The number of followers the user has, which can be viewed as their "social influence."  
* **elite**: The years the user earned "Yelp Elite" status (recorded as a string or array of years).  
* **friends**: A list of the user’s friend connections on the platform (containing a sequence of other user\_ids).

**User Feedback Metrics:**

* **useful**: Total "Useful" votes the user’s reviews have received from others.  
* **funny**: Total "Funny" votes received.  
* **cool**: Total "Cool" votes received.

**Compliments:**

Yelp’s unique badge system. Includes fields like   
compliment\_hot,   
compliment\_more,   
compliment\_profile,   
compliment\_cute,   
compliment\_list,   
compliment\_note,   
compliment\_plain,   
compliment\_cool,   
compliment\_funny,   
compliment\_writer,   
and compliment\_photos.   
These represent various types of specific appreciation bestowed by other users.

### ---

**2\. data/item\_subset.json (Entity Item Subset \- Yelp Only)**

This file records information for the businesses (entities) reviewed by active users (similarly, source and type have been removed):

* **item\_id**: The unique identifier for the business or location (Primary Key).  
* **name**: The name of the business or establishment.  
* **stars**: The overall average rating of the business on the platform ($1.0$ to $5.0$).  
* **review\_count**: The total number of reviews the business has received.  
* **is\_open**: An operational status flag ($1$ indicates the business is open, $0$ indicates it is permanently closed).

**Geographic Information:**

* **address**: Street address.  
* **city**: The city where it is located.  
* **state**: The abbreviation of the state or province.  
* **postal\_code**: ZIP/Postal code.  
* **latitude & longitude**: Geographical coordinates, useful for calculating distance or plotting on a map.

**Additional Features:**

* **categories**: Tagged business categories (e.g., "Restaurants, Italian, Coffee & Tea"). These are crucial for recommendation systems to capture Item features.  
* **hours**: Specific operating hours for the seven days of the week (e.g., "Monday": "8:0-22:0").  
* **attributes**: A complex dictionary describing the business's facilities and conditions, such as BusinessParking (availability of parking),   
  WiFi (internet access),   
  GoodForKids (child-friendliness),   
  and RestaurantsPriceRange2 (price bracket).   
  This provides a rich source of NLP or structured features.

---

### **3\. data/review\_subset.json (Interaction/Review Subset \- Yelp Only)**

This file serves as the **bridge (Edge / Interaction)** connecting user\_subset.json and item\_subset.json. It captures the core relationship: "which user left what feedback for which business."

**Entity Relationship Keys (Foreign Keys):**

* **review\_id**: The unique identifier (String) for the review itself. This ID is used for distributed storage or when retrieving a specific individual review.  
* **user\_id**: The ID of the author who posted the review. This maps directly to the user\_id in user\_subset.json, allowing you to retrieve user-specific features like fan count or activity levels.  
* **item\_id**: The ID of the business being reviewed. This maps directly to the item\_id in item\_subset.json to pull business-specific features like coordinates or categories.

**Review Content (Content & Sentiment):**

* **stars**: The star rating the user gave the business (floating-point value between $1.0$ and $5.0$). This is typically the **Target Label (Ground Truth)** used when training "Rating Prediction" models.  
* **text**: The actual text content of the review. This is the most valuable **unstructured data** in the set, suitable for Sentiment Analysis, keyword extraction, or training LLM-based Recommendation Agents.  
* **date**: The timestamp of when the review was published (Format: "YYYY-MM-DD HH:MM:SS"). This can be used to analyze user temporal preferences or seasonal trends. (Note: This field—converted to a Unix Timestamp—was used to perform the chronological split for the first 80% of the training set.)

**Social Feedback:**

These three fields represent the number of "resonance votes" cast by **other Yelp users** regarding the quality of this specific review:

* **useful**: The number of users who voted the review as "Useful."  
* **funny**: The number of users who voted the review as "Funny/Hilarious."  
* **cool**: The number of users who voted the review as "Cool."

**Note:** These three metrics are frequently used in modern models as a **Review Quality Weight**. Reviews with a higher "useful" count often provide more representative and reliable textual information.

