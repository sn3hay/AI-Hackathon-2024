import os
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re  # Import regex for name extraction
import tempfile  # Import tempfile for temporary file handling
from PIL import Image
from dotenv import load_dotenv
from groq import Groq        
load_dotenv('groqapi.env')

client = Groq(
    api_key=os.environ['GROQ_API_KEY'],
)

# Initialize Streamlit session state
if 'memory' not in st.session_state:
    st.session_state['memory'] = []  # Use a list for memory
if 'user_name' not in st.session_state:
    st.session_state['user_name'] = None  # To store the user's name

# Initialize embeddings model and vector store
embeddings_model = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
vector_store = None

# Store previous question embeddings
question_embeddings = []

# Prompt templates for LLM
prompt_with_context_template = """ Analyze the following context and answer the question based only on the following context:
{context}
Question: {question}
if the question is about career path and if he is in undergrad suggest top undergrad, top masters degrees from western new york community or if the user is from k-12 grade suggest the top undergrad colleges from western new york community.
"""
prompt_without_context_template = """You are the best assistant so answer accordingly to the question
Question: {question}
if the question is about career path and if he is in undergrad suggest top undergrad, top masters degrees from western new york community or if the user is from k-12 grade suggest the top undergrad colleges from western new york community.
"""
prompt_with_context = PromptTemplate.from_template(prompt_with_context_template)
prompt_without_context = PromptTemplate.from_template(prompt_without_context_template)

# Function to load, split PDFs, and store in vector store
def process_documents(uploaded_files):
    global vector_store
    all_docs = []
    for uploaded_file in uploaded_files:
        # Save the uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name  # Get the temporary file path

        # Load the PDF using PyPDFLoader
        loader = PyPDFLoader(temp_file_path)  # Use the temporary file path
        pages = loader.load_and_split()

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(pages)
        all_docs.extend(docs)

    # Create or update the vector store
    if vector_store is None:
        vector_store = Chroma.from_documents(all_docs, embeddings_model)
    else:
        vector_store.add_documents(all_docs)
    
    return f"Uploaded {len(uploaded_files)} files and indexed {len(all_docs)} chunks."

# Function to handle question answering with RAG and maintain chat history
def answer_question(question):
    global vector_store, question_embeddings
    
    # Set up retriever and LLM
    retriever = vector_store.as_retriever() if vector_store else None
    llm = ChatOllama(model="llama3:latest", verbose=True)

    if retriever:
        # Define the RAG chain with document context
        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt_with_context
            | llm
            | StrOutputParser()
        )
        # Process user question through RAG chain with context
        answer = chain.invoke(question).capitalize()
    else:
        # Define the RAG chain without document context
        chain = (
            {"question": RunnablePassthrough()}
            | prompt_without_context
            | llm
            | StrOutputParser()
        )
        # Process user question through RAG chain without context
        answer = chain.invoke(question).capitalize()

    # Store the question and answer in memory
    st.session_state['memory'].append({"user": question, "bot": answer})
    
    # Encode the current question and store its embedding
    current_question_embedding = embeddings_model.embed_query(question)
    question_embeddings.append(current_question_embedding)
    
    # Find related questions
    related_question = "No related questions found."
    if question_embeddings:
        # Compute similarity between current question and previous questions
        similarities = cosine_similarity([current_question_embedding], question_embeddings)
        related_idx = np.argmax(similarities)
        if similarities[0][related_idx] > 0.5:
            related_question = st.session_state['memory'][related_idx]['user']  # Retrieve the related question 
    
    return answer, related_question

# Function to clear the vector store and memory
def clear_documents():
    global vector_store
    if vector_store is not None:
        vector_store.delete_collection()
        vector_store = None
    st.session_state['memory'].clear()  # Clear conversation memory
    st.session_state['user_name'] = None  # Clear stored user name
    return "Document collection and memory cleared."
    
course_inventory = [
    {"name": "Introduction to Prompt Engineering", "category": "Prompt Engineering"},
    {"name": "Advanced Generative AI Techniques", "category": "Generative AI"},
    {"name": "Machine Learning Fundamentals", "category": "Machine Learning"},
    {"name": "Cybersecurity Essentials", "category": "Cybersecurity"},
    {"name": "Computer Science Principles", "category": "Computer Science"},
    {"name": "Deep Learning and Neural Networks", "category": "Machine Learning"},
    {"name": "Ethical Hacking and Penetration Testing", "category": "Cybersecurity"},
    {"name": "Natural Language Processing", "category": "Generative AI"},
    {"name": "Data Structures and Algorithms", "category": "Computer Science"},
    {"name": "AI for Business Applications", "category": "Generative AI"},
    {"name": "Network Security", "category": "Cybersecurity"},
    {"name": "Reinforcement Learning", "category": "Machine Learning"},
    {"name": "Software Engineering Practices", "category": "Computer Science"},
    {"name": "AI Ethics and Society", "category": "Generative AI"},
    {"name": "Cloud Security Fundamentals", "category": "Cybersecurity"},
    {"name": "Advanced Prompt Engineering", "category": "Prompt Engineering"},
    {"name": "Computer Vision", "category": "Generative AI"},
    {"name": "Blockchain and Cybersecurity", "category": "Cybersecurity"},
    {"name": "Data Mining and Big Data Analytics", "category": "Machine Learning"},
    {"name": "Operating Systems", "category": "Computer Science"}
]

course_details = [f"{course['name']} ({course['category']})" for course in course_inventory]

def answer_question_course(question):
    try:
        completion = client.chat.completions.create(model="llama3-70b-8192",
        messages=[
                {
                "role": "system", "content":     
                f"""You are a knowledgeable course advisor for the Buffalo community. 
                You should only provide information about the courses from the list. 
                Given the list of courses available: {', '.join(course_details)}, please provide details about a specific course. 
                If the course is not in the list, respond that the course is not available. If the user asks about any other topic, 
                respond accordingly without adding any course-related prompts. Also if you found match with  multiple courses rank and show                          percentage of match to the user  based on the similarity of skills provided."""

                },
                {"role": "user", "content": question }],
                temperature=0.6,
                max_tokens=1500,
                top_p=1,
                stream=False,
                stop=None,
        )

        
        st.session_state['memory'].append({"user": question, "bot": completion.choices[0].message.content})
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Error generating response: {e}")
        return "Sorry, I couldn't generate a response."


job_inventory = [
    {
        "title": "Data Scientist",
        "company": "M&T Bank",
        "salary": "$90,000 - $120,000",
        "description": "Responsible for building predictive models and analyzing complex datasets to support business decisions."
    },
    {
        "title": "Machine Learning Engineer",
        "company": "Moog Inc.",
        "salary": "$100,000 - $130,000",
        "description": "Develops and implements machine learning algorithms to improve product performance and operations."
    },
    {
        "title": "Cybersecurity Analyst",
        "company": "BlueCross BlueShield of WNY",
        "salary": "$85,000 - $110,000",
        "description": "Monitors and defends network security, ensuring compliance with security standards and protocols."
    },
    {
        "title": "Software Developer",
        "company": "ACV Auctions",
        "salary": "$80,000 - $110,000",
        "description": "Designs, develops, and maintains software applications to improve auction platform services."
    },
    {
        "title": "Cloud Engineer",
        "company": "Delaware North",
        "salary": "$95,000 - $125,000",
        "description": "Designs and implements cloud solutions, ensuring efficient operation of cloud-based infrastructure."
    },
    {
        "title": "AI Researcher",
        "company": "CUBRC",
        "salary": "$105,000 - $140,000",
        "description": "Conducts research in artificial intelligence to develop innovative solutions for data analysis and automation."
    },
    {
        "title": "IT Support Specialist",
        "company": "Rich Products Corporation",
        "salary": "$60,000 - $80,000",
        "description": "Provides technical support to employees, troubleshooting hardware and software issues."
    },
    {
        "title": "DevOps Engineer",
        "company": "Seneca Gaming Corporation",
        "salary": "$90,000 - $115,000",
        "description": "Automates and optimizes development pipelines, ensuring smooth integration and deployment of software."
    },
    {
        "title": "Network Engineer",
        "company": "Kaleida Health",
        "salary": "$85,000 - $110,000",
        "description": "Designs, implements, and manages network infrastructure for hospital operations and healthcare services."
    },
    {
        "title": "Front-End Developer",
        "company": "Liazon",
        "salary": "$75,000 - $100,000",
        "description": "Builds and maintains user interfaces for web applications, ensuring a seamless user experience."
    },
    {
        "title": "Database Administrator",
        "company": "Independent Health",
        "salary": "$85,000 - $115,000",
        "description": "Manages and optimizes databases, ensuring data integrity and performance for healthcare applications."
    },
    {
        "title": "Business Analyst",
        "company": "SofTrek Corporation",
        "salary": "$70,000 - $95,000",
        "description": "Analyzes business processes to recommend improvements and support data-driven decision making."
    },
    {
        "title": "Cloud Security Architect",
        "company": "Hodgson Russ LLP",
        "salary": "$110,000 - $140,000",
        "description": "Designs and implements secure cloud architectures to protect client data and legal operations."
    },
    {
        "title": "Data Engineer",
        "company": "Buffalo Medical Group",
        "salary": "$90,000 - $120,000",
        "description": "Builds and manages data pipelines to support data analytics and healthcare decision-making."
    },
    {
        "title": "Cybersecurity Specialist",
        "company": "Harmac Medical Products",
        "salary": "$85,000 - $115,000",
        "description": "Ensures the security of medical products and systems, identifying and mitigating potential cyber threats."
    },
    {
        "title": "Product Manager",
        "company": "Launch NY",
        "salary": "$100,000 - $130,000",
        "description": "Leads product development, managing cross-functional teams to deliver innovative solutions for startups."
    },
    {
        "title": "Full-Stack Developer",
        "company": "Bak USA",
        "salary": "$85,000 - $110,000",
        "description": "Develops both front-end and back-end systems for innovative tech products and web applications."
    },
    {
        "title": "AI Solutions Architect",
        "company": "Pegula Sports and Entertainment",
        "salary": "$115,000 - $150,000",
        "description": "Designs AI solutions for sports analytics and fan engagement, leveraging advanced machine learning models."
    },
    {
        "title": "IT Consultant",
        "company": "Freed Maxick CPAs, P.C.",
        "salary": "$90,000 - $120,000",
        "description": "Advises businesses on IT strategies, providing solutions for optimizing technology and business processes."
    },
    {
        "title": "Project Manager",
        "company": "Northland Workforce Training Center",
        "salary": "$85,000 - $110,000",
        "description": "Oversees project planning and execution, ensuring timely delivery of workforce training programs."
    }
]

job_details = [f"{job['title']} at {job['company']} with a salary range of {job['salary']}. {job['description']}" for job in job_inventory]


def answer_question_job(question):
    try:
        completion = client.chat.completions.create(model="llama3-70b-8192",
        messages=[
                {
                "role": "system", "content":     
                f"""You are Jordan, a helpful job advisor for Western New York. 
                You should only provide information about the jobs available in our catalog. 
                Given the list of jobs available: {', '.join(job_details)}, please provide details about a specific job(description , company and                   salary in dollars). If the job is not in the list, respond that the job is not available. If the user asks about any other topic, 
                respond accordingly without adding any job-related prompts. Also if you found match with multiple jobs  rank and show percentage of                  match to the user  based on the similarity of skills provided."""

                },
                {"role": "user", "content": question }],
                temperature=0.6,
                max_tokens=1500,
                top_p=1,
                stream=False,
                stop=None,
        )

        
        st.session_state['memory'].append({"user1": question, "bot1": completion.choices[0].message.content})
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Error generating response: {e}")
        return "Sorry, I couldn't generate a response."


job_postings = [
    {
        "title": "Software Engineer",
        "company": "TechCorp",
        "location": "Erie, PA",
        "description": "Looking for an experienced software engineer skilled in Python, AWS, and microservices.",
        "skills": ["Python", "AWS", "Microservices"],
        "salary": "$95,000 - $115,000",
        "posted": "2 days ago"
    },
    {
        "title": "Data Scientist",
        "company": "Data Solutions Inc.",
        "location": "Erie, PA",
        "description": "Seeking a data scientist with experience in machine learning, TensorFlow, and big data.",
        "skills": ["Machine Learning", "TensorFlow", "Big Data"],
        "salary": "$110,000 - $130,000",
        "posted": "5 days ago"
    },
    {
        "title": "Web Developer",
        "company": "Web Innovators",
        "location": "Erie, PA",
        "description": "Join our team as a web developer with expertise in JavaScript, React, and CSS.",
        "skills": ["JavaScript", "React", "CSS"],
        "salary": "$80,000 - $100,000",
        "posted": "1 week ago"
    },
    {
        "title": "DevOps Engineer",
        "company": "Cloud Solutions",
        "location": "Erie, PA",
        "description": "Hiring a DevOps engineer with experience in CI/CD, Docker, and Kubernetes.",
        "skills": ["CI/CD", "Docker", "Kubernetes"],
        "salary": "$105,000 - $125,000",
        "posted": "3 days ago"
    },
    {
        "title": "Frontend Developer",
        "company": "CreativeTech",
        "location": "Erie, PA",
        "description": "Looking for a frontend developer skilled in HTML, CSS, and JavaScript frameworks.",
        "skills": ["HTML", "CSS", "JavaScript", "React"],
        "salary": "$75,000 - $90,000",
        "posted": "4 days ago"
    },
    {
        "title": "Backend Developer",
        "company": "APISolutions",
        "location": "Erie, PA",
        "description": "Seeking a backend developer with Node.js, Python, and database management experience.",
        "skills": ["Node.js", "Python", "SQL", "MongoDB"],
        "salary": "$90,000 - $110,000",
        "posted": "6 days ago"
    },
    {
        "title": "Systems Administrator",
        "company": "NetSec Corp",
        "location": "Erie, PA",
        "description": "Systems admin needed with expertise in Linux, Windows Server, and cloud infrastructure.",
        "skills": ["Linux", "Windows Server", "AWS"],
        "salary": "$85,000 - $105,000",
        "posted": "2 days ago"
    },
    {
        "title": "Mobile App Developer",
        "company": "AppSoft",
        "location": "Erie, PA",
        "description": "Hiring a mobile app developer with experience in Android and iOS app development.",
        "skills": ["Android", "iOS", "Flutter", "React Native"],
        "salary": "$100,000 - $120,000",
        "posted": "1 week ago"
    },
    {
        "title": "IT Support Specialist",
        "company": "SupportPros",
        "location": "Erie, PA",
        "description": "IT support role open for candidates with experience in hardware troubleshooting and helpdesk support.",
        "skills": ["Troubleshooting", "Helpdesk", "Hardware"],
        "salary": "$55,000 - $70,000",
        "posted": "2 days ago"
    },
    {
        "title": "Cybersecurity Analyst",
        "company": "SecureNet",
        "location": "Erie, PA",
        "description": "Seeking a cybersecurity analyst with experience in threat detection, incident response, and network security.",
        "skills": ["Threat Detection", "Incident Response", "Network Security"],
        "salary": "$95,000 - $115,000",
        "posted": "3 days ago"
    },
    {
        "title": "UX/UI Designer",
        "company": "DesignHub",
        "location": "Erie, PA",
        "description": "Looking for a UX/UI designer with experience in wireframing, prototyping, and user research.",
        "skills": ["Wireframing", "Prototyping", "User Research"],
        "salary": "$70,000 - $90,000",
        "posted": "4 days ago"
    },
    {
        "title": "Data Scientist",
        "company": "M&T Bank",
        "location": "Erie, PA",
        "description": "Responsible for building predictive models and analyzing complex datasets to support business decisions.",
        "skills": ["Predictive Modeling", "Data Analysis", "Business Intelligence"],
        "salary": "$90,000 - $120,000",
        "posted": "2 days ago"
    },
    {
        "title": "Machine Learning Engineer",
        "company": "Moog Inc.",
        "location": "Erie, PA",
        "description": "Develops and implements machine learning algorithms to improve product performance and operations.",
        "skills": ["Machine Learning", "Algorithm Development", "Data Engineering"],
        "salary": "$100,000 - $130,000",
        "posted": "5 days ago"
    },
    {
        "title": "Cybersecurity Analyst",
        "company": "BlueCross BlueShield of WNY",
        "location": "Erie, PA",
        "description": "Monitors and defends network security, ensuring compliance with security standards and protocols.",
        "skills": ["Network Security", "Threat Analysis", "Compliance"],
        "salary": "$85,000 - $110,000",
        "posted": "1 week ago"
    },
    {
        "title": "Software Developer",
        "company": "ACV Auctions",
        "location": "Erie, PA",
        "description": "Designs, develops, and maintains software applications to improve auction platform services.",
        "skills": ["Software Development", "Application Maintenance", "Java"],
        "salary": "$80,000 - $110,000",
        "posted": "1 week ago"
    },
    {
        "title": "Cloud Engineer",
        "company": "Delaware North",
        "location": "Erie, PA",
        "description": "Designs and implements cloud solutions, ensuring efficient operation of cloud-based infrastructure.",
        "skills": ["Cloud Computing", "Infrastructure Design", "AWS"],
        "salary": "$95,000 - $125,000",
        "posted": "1 week ago"
    },
    {
        "title": "AI Researcher",
        "company": "CUBRC",
        "location": "Erie, PA",
        "description": "Conducts research in artificial intelligence to develop innovative solutions for data analysis and automation.",
        "skills": ["Artificial Intelligence", "Research", "Data Analysis"],
        "salary": "$105,000 - $140,000",
        "posted": "2 weeks ago"
    },
    {
        "title": "IT Support Specialist",
        "company": "Rich Products Corporation",
        "location": "Erie, PA",
        "description": "Provides technical support to employees, troubleshooting hardware and software issues.",
        "skills": ["Technical Support", "Troubleshooting", "Customer Service"],
        "salary": "$60,000 - $80,000",
        "posted": "3 weeks ago"
    },
    {
        "title": "DevOps Engineer",
        "company": "Seneca Gaming Corporation",
        "location": "Erie, PA",
        "description": "Automates and optimizes development pipelines, ensuring smooth integration and deployment of software.",
        "skills": ["DevOps", "Pipeline Automation", "CI/CD"],
        "salary": "$90,000 - $115,000",
        "posted": "2 weeks ago"
    },
    {
        "title": "Network Engineer",
        "company": "Kaleida Health",
        "location": "Erie, PA",
        "description": "Designs, implements, and manages network infrastructure for hospital operations and healthcare services.",
        "skills": ["Network Design", "Infrastructure Management", "Healthcare IT"],
        "salary": "$85,000 - $110,000",
        "posted": "1 week ago"
    },
    {
        "title": "Front-End Developer",
        "company": "Liazon",
        "location": "Erie, PA",
        "description": "Builds and maintains user interfaces for web applications, ensuring a seamless user experience.",
        "skills": ["HTML", "CSS", "JavaScript"],
        "salary": "$75,000 - $100,000",
        "posted": "1 week ago"
    },
    {
        "title": "Database Administrator",
        "company": "Independent Health",
        "location": "Erie, PA",
        "description": "Manages and optimizes databases, ensuring data integrity and performance for healthcare applications.",
        "skills": ["Database Management", "SQL", "Data Integrity"],
        "salary": "$85,000 - $115,000",
        "posted": "2 weeks ago"
    },
    {
        "title": "Business Analyst",
        "company": "SofTrek Corporation",
        "location": "Erie, PA",
        "description": "Analyzes business processes to recommend improvements and support data-driven decision making.",
        "skills": ["Business Analysis", "Process Improvement", "Data-Driven Decision Making"],
        "salary": "$70,000 - $95,000",
        "posted": "1 week ago"
    },
    {
        "title": "Cloud Security Architect",
        "company": "Hodgson Russ LLP",
        "location": "Erie, PA",
        "description": "Designs and implements secure cloud architectures to protect client data and legal operations.",
        "skills": ["Cloud Security", "Architecture Design", "Risk Management"],
        "salary": "$110,000 - $140,000",
        "posted": "3 weeks ago"
    },
    {
        "title": "Data Engineer",
        "company": "Buffalo Medical Group",
        "location": "Erie, PA",
        "description": "Builds and manages data pipelines to support data analytics and healthcare decision-making.",
        "skills": ["Data Engineering", "Pipeline Management", "Analytics Support"],
        "salary": "$90,000 - $120,000",
        "posted": "1 month ago"
    },
    {
        "title": "Cybersecurity Specialist",
        "company": "Harmac Medical Products",
        "location": "Erie, PA",
        "description": "Ensures the security of medical products and systems, identifying and mitigating potential cyber threats.",
        "skills": ["Cybersecurity", "Threat Mitigation", "Medical Product Security"],
        "salary": "$85,000 - $115,000",
        "posted": "1 month ago"
    },
    {
        "title": "Product Manager",
        "company": "Launch NY",
        "location": "Erie, PA",
        "description": "Leads product development, managing cross-functional teams to deliver innovative solutions for startups.",
        "skills": ["Product Management", "Cross-Functional Leadership", "Innovation"],
        "salary": "$100,000 - $130,000",
        "posted": "1 month ago"
    },
    {
        "title": "Full-Stack Developer",
        "company": "Bak USA",
        "location": "Erie, PA",
        "description": "Develops both front-end and back-end systems for innovative tech products and web applications.",
        "skills": ["Full-Stack Development", "JavaScript", "React"],
        "salary": "$85,000 - $110,000",
        "posted": "1 month ago"
    },
    {
        "title": "AI Solutions Architect",
        "company": "Pegula Sports and Entertainment",
        "location": "Erie, PA",
        "description": "Designs AI solutions for sports analytics and fan engagement, leveraging advanced machine learning models.",
        "skills": ["AI Solutions", "Sports Analytics", "Machine Learning"],
        "salary": "$115,000 - $150,000",
        "posted": "1 month ago"
    },
    {
        "title": "IT Consultant",
        "company": "Freed Maxick CPAs, P.C.",
        "location": "Erie, PA",
        "description": "Advises businesses on IT strategies, providing solutions for optimizing technology and business processes.",
        "skills": ["IT Consulting", "Technology Optimization", "Business Processes"],
        "salary": "$90,000 - $120,000",
        "posted": "2 months ago"
    },
    {
        "title": "Project Manager",
        "company": "Northland Workforce Training Center",
        "location": "Erie, PA",
        "description": "Oversees project planning and execution, ensuring timely delivery of workforce training programs.",
        "skills": ["Project Management", "Planning", "Execution"],
        "salary": "$85,000 - $110,000",
        "posted": "2 months ago"
    }
]

def display_job_postings(jobs, title):
    st.subheader(title)
    for job in jobs:
        st.write(f"### {job['title']} at {job['company']}")
        st.write(f"**Location**: {job['location']}")
        st.write(f"**Skills**: {', '.join(job['skills'])}")
        st.write(f"**Posted**: {job['posted']}")
        st.write(f"**Salary**: {job['salary']}")
        st.write(f"*{job['description']}*")
        st.markdown("---")
        
# Responses history
if 'responses' not in st.session_state:
    st.session_state['responses'] = []


def display_images(image_paths, captions):
    for path, caption in zip(image_paths, captions):
        image = Image.open(path)
        st.image(image, caption=caption)
        
# Sidebar Navigation Options
st.sidebar.title("Options")
options = st.sidebar.radio(
    "Navigate", 
    ["Job Openings Trend in Community", "Career Pathway Based on Skills", "Recommended Programs","Employer Job postings and Recommended Jobs "]
)


# Option routing
if options == "Career Pathway Based on Skills":
    st.title("Career Pathway Based on Skills")

    # User input
    uploaded_files = st.file_uploader("📤Upload PDFs", type=["pdf"], accept_multiple_files=True)

    if st.button("📤Upload and Process"):
        if uploaded_files:
            status_message = process_documents(uploaded_files)
            st.success(status_message)
        else:
            st.error("Please upload at least one PDF file.")

    # User question input
    ui_prompt = st.chat_input("Enter your question")
    if ui_prompt:
        answer, related_question = answer_question(ui_prompt)

        # Display the answer and related question
        #st.text_area("Answer", value=answer, height=100)
        #st.text_area("Related Question", value=related_question, height=50)

    # Display chat history
    if st.session_state['memory']:
        chat_history = "\n\n".join([f"User: {item['user']}\nBot: {item['bot']}" for item in st.session_state['memory']])
        st.text_area("Chat History", value=chat_history, height=300)

    # Button to clear documents and memory
    if st.button("Clear Document Collection and Memory"):
        clear_documents()
        st.success("Document collection and memory cleared.")


            
elif options == "Job Openings Trend in Community":

    st.title("Job Openings Trend in Community")
    # Display initial plots for job growth
    st.subheader("Job Growth Trends")
    image_paths = [
        "sample1.png",  # Replace with actual image paths
        "sample2.png",
        "sample3.png"
    ]
    captions = [
        "Growth of Software Jobs in WNY upto 10 Years",
        "Growth of Software Jobs in WNY upto 10 Years",
        "Growth of Software Jobs in WNY upto 10 Years"
    ]
    display_images(image_paths, captions)

    # Display salary range for job titles
    st.subheader("Salary Range for Different Job Titles")
    salary_range_image_path = "salary-range.png"  # Replace with the actual image path
    st.image(salary_range_image_path, caption="Salary Range for Different Job Titles in WNY with min, max and avg salaries")

    # Dropdown for specific job role growth plots
    st.subheader("Growth of Software Jobs in Erie by Role")
    job_roles = [
        "Web Developers",
        "Applications Developers",
        "Network and Computer Systems Administrators",
        "Information Security Analysts",
        "Computer Systems Analysts",
        "Computer and Information Research Scientists",
        "Computer Programmers",
        "Network and computer system administrator"
    ]
    
    selected_role = st.selectbox("Select a Job Role", job_roles)

    # Display growth plot based on selected role
    if selected_role == "Web Developers":
        web_dev_growth_path = "webdevs.png"  # Replace with actual image path
        st.image(web_dev_growth_path, caption="Growth of Software Jobs (Web Developers) in WNY")

    elif selected_role == "Applications Developers":
        app_dev_growth_path = "sdes.png"  # Replace with actual image path
        st.image(app_dev_growth_path, caption="Growth of Software Jobs (Applications Developers) in WNY")

    elif selected_role == "Network and Computer Systems Administrators":
        network_admin_growth_path = "net-comp-sys-admins.png"  # Replace with actual image path
        st.image(network_admin_growth_path, caption="Growth of Software Jobs (Network and Computer Systems Administrators) in WNY")

    elif selected_role == "Information Security Analysts":
        info_sec_growth_path = "infosec.png"  # Replace with actual image path
        st.image(info_sec_growth_path, caption="Growth of Software Jobs (Information Security Analysts) in WNY")

    elif selected_role == "Computer Systems Analysts":
        sys_analyst_growth_path = "compsys.png"  # Replace with actual image path
        st.image(sys_analyst_growth_path, caption="Growth of Software Jobs (Computer Systems Analysts) in WNY")

    elif selected_role == "Computer and Information Research Scientists":
        info_research_growth_path = "comp-scientists.png"  # Replace with actual image path
        st.image(info_research_growth_path, caption="Growth of Software Jobs (Computer and Information Research Scientists) in WNY")

    elif selected_role == "Computer Programmers":
        programmer_growth_path = "comp-programmers.png"  # Replace with actual image path
        st.image(programmer_growth_path, caption="Growth of Software Jobs (Computer Programmers) in WNY")
        
    elif selected_role == "Network and computer system administrator":
        programmer_growth_path = "net-comp-sys-admins.png"  # Replace with actual image path
        st.image(programmer_growth_path, caption="Growth of Software Jobs (Network and computer system administrator) in WNY")


elif options == "Recommended Programs":
    st.subheader("Recommended Programs (non degree training programs) based on your skills")
    #st.write("")

    
    ui_prompt = st.chat_input("💬 Share your skills, and we'll recommend tailored non-degree training courses to help you excel.")
    if ui_prompt:
        answer = answer_question_course(ui_prompt)

        st.session_state['responses'].append(("user", ui_prompt))
        st.session_state['responses'].append(("bot", answer))
        
        
elif options =="Employer Job postings and Recommended Jobs ":
    st.subheader("Employer Job Postings")
    display_job_postings(job_postings, "Latest:")
    ui_prompt = st.chat_input("💬 Enter your skills to receive personalized job recommendations.")
    
    if ui_prompt:
        answer = answer_question_job(ui_prompt)

        st.session_state['responses'].append(("user", ui_prompt))
        st.session_state['responses'].append(("bot", answer))

# Display chat history
for role, message in st.session_state['responses']:
    if role == 'user':
        st.write(f"**You:** {message}")
    else:
        st.write(f"**Bot:** {message}")


