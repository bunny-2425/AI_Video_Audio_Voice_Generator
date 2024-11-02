import streamlit as st
import openai
import requests
import json

def main():
    st.title("Azure OpenAI GPT-4o Connectivity Test") 
    
    # Azure OpenAI connection details
    # Here, we define the API key and endpoint URL for connecting to Azure OpenAI.
    azure_openai_key = "Enter API KEY"  # Replace with your actual key. if you don't have one, get from Azure or from Community https://curious.pm
    azure_openai_endpoint = "Enter endpoint URL"  # Replace with your actual endpoint URL
    
    # Button to initiate the connection and request
    # User clicks this button to initiate the request to Azure OpenAI.
    if st.button("Connect and Get Response"):
        
        # Check if both the key and endpoint are provided
        # Ensure that the key and endpoint are not empty before proceeding.
        if azure_openai_key and azure_openai_endpoint:
            try:
                # Setting up headers for the API request
                # Define the headers needed for the API request, including the API key for authentication.
                headers = {
                    "Content-Type": "application/json",  # Specifies that we are sending JSON data
                    "api-key": azure_openai_key  # The API key for authentication
                }
                
                # Data to be sent to Azure OpenAI
                # Define the payload, which includes the message prompt and token limit.
                # **** This is where you can customize the message prompt and token limit. ****
                data = {
                    "messages": [{"role": "user", "content": "Hello, Azure OpenAI!"}],  # The message we want the model to respond to
                    "max_tokens": 50  # Limit the response length
                }
                
                # Making the POST request to the Azure OpenAI endpoint
                # Send the request to the Azure OpenAI endpoint using the defined headers and data.
                response = requests.post(azure_openai_endpoint, headers=headers, json=data)
                
                # Check if the request was successful
                # Handle the response, checking the status and displaying the result.
                if response.status_code == 200:
                    result = response.json()  # Parse the JSON response
                    st.success(result["choices"][0]["message"]["content"].strip())  # Display the response content from the AI
                else:
                    # Handle errors if the request was not successful
                    st.error(f"Failed to connect or retrieve response: {response.status_code} - {response.text}")
            except Exception as e:
                # Handle any exceptions that occur during the request
                st.error(f"Failed to connect or retrieve response: {str(e)}")
        else:
            # Warn the user if key or endpoint is missing
            st.warning("Please enter all the required details.")

if __name__ == "__main__":
    main()