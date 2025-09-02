import PyPDF2
import re
import google.generativeai as genai
from typing import Dict, List, Tuple
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class ResumeParserAgent:
    def summarize_github_for_gemini(self, github_analysis: dict) -> str:
        """Summarizes GitHub analysis for Gemini prompt."""
        if not github_analysis or 'error' in github_analysis:
            return "No valid GitHub data."
        summary = f"GitHub Profile: {github_analysis.get('github_url', 'N/A')}, Total Public Repos: {github_analysis.get('total_repos', 'N/A')}\n"
        for repo in github_analysis.get('relevant_repos', []):
            summary += f"Project: {repo['name']}\nDescription: {repo['description']}\nLanguages: {', '.join(repo['languages'])}\nStars: {repo['stars']}, Forks: {repo['forks']}\nContributors: {', '.join(repo['contributors'])}\nIssues: {repo['issues']}, PRs: {repo['pulls']}\nLast Commit: {repo['last_commit']}\nREADME: {repo['readme'][:200].replace(chr(10), ' ')}\n---\n"
        return summary

    def combined_gemini_score(self, job_description: str, resume_text: str, github_summary: str) -> dict:
        """Uses Gemini to score candidate based on both resume and GitHub data."""
        prompt = (
            f"Analyze the following candidate for suitability for the job. Consider both resume and GitHub profile data.\n\n"
            f"JOB DESCRIPTION:\n{job_description}\n\n"
            f"RESUME:\n{resume_text}\n\n"
            f"GITHUB PROFILE DATA:\n{github_summary}\n\n"
            "Provide a JSON response with:\n{\n  'combined_suitability_score': 0-100,\n  'github_score': 0-100,\n  'resume_score': 0-100,\n  'summary': 'text summary comparing resume and GitHub',\n  'recommendation': 'text recommendation'\n}\n"
        )
        try:
            response = self.model.generate_content(prompt)
            json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return {
                    'combined_suitability_score': 0,
                    'github_score': 0,
                    'resume_score': 0,
                    'summary': 'Failed to parse Gemini response.',
                    'recommendation': ''
                }
        except Exception as e:
            return {
                'combined_suitability_score': 0,
                'github_score': 0,
                'resume_score': 0,
                'summary': f'Error calling Gemini: {str(e)}',
                'recommendation': ''
            }
    def extract_github_profile(self, resume_text: str) -> str:
        """
        Extracts GitHub username from resume text.
        Handles full links, 'Github: username', and partial links in descriptions.
        Returns the username if found, else None.
        """
        # Try full GitHub profile link
        match = re.search(r"https?://github\.com/([A-Za-z0-9_-]+)", resume_text)
        if match:
            return match.group(1)
        # Try 'Github: username' or 'GitHub username'
        match = re.search(r"github\s*[:\-]?\s*([A-Za-z0-9_-]+)", resume_text, re.IGNORECASE)
        if match:
            return match.group(1)
        # Try any github.com/username in project descriptions
        matches = re.findall(r"github\.com/([A-Za-z0-9_-]+)", resume_text)
        if matches:
            return matches[0]
        return None

    def analyze_github_projects(self, github_username: str, job_keywords: list) -> dict:
        """
        Uses PyGithub to fetch user repos and check relevance to job keywords.
        Returns a summary dict.
        """
        from github import Github
        github_token = os.getenv("GITHUB_TOKEN")
        if github_token:
            g = Github(github_token)
        else:
            g = Github()  # Falls back to unauthenticated if no token
        try:
            user = g.get_user(github_username)
            repos = user.get_repos()
            relevant_repos = []
            for repo in repos:
                # Get recent commit activity (last commit date)
                try:
                    commits = repo.get_commits()
                    last_commit = commits[0].commit.committed_date if commits.totalCount > 0 else None
                except Exception:
                    last_commit = None
                # Get primary languages used
                try:
                    languages = list(repo.get_languages().keys())
                except Exception:
                    languages = []
                # Get contributors
                try:
                    contributors = [contrib.login for contrib in repo.get_contributors()]
                except Exception:
                    contributors = []
                # Get README content
                try:
                    readme = repo.get_readme().decoded_content.decode('utf-8')
                except Exception:
                    readme = ''
                # Get issues and PRs
                try:
                    issues = repo.get_issues(state='all').totalCount
                except Exception:
                    issues = 0
                try:
                    pulls = repo.get_pulls(state='all').totalCount
                except Exception:
                    pulls = 0
                # Get repo size and file count
                try:
                    size = repo.size  # in KB
                except Exception:
                    size = 0
                try:
                    contents = repo.get_contents("")
                    file_count = len(contents)
                except Exception:
                    file_count = 0
                repo_data = {
                    'name': repo.name,
                    'description': repo.description,
                    'language': repo.language,
                    'languages': languages,
                    'topics': repo.get_topics(),
                    'url': repo.html_url,
                    'stars': repo.stargazers_count,
                    'forks': repo.forks_count,
                    'created_at': str(repo.created_at),
                    'updated_at': str(repo.updated_at),
                    'last_commit': str(last_commit) if last_commit else 'No commits',
                    'contributors': contributors,
                    'readme': readme,
                    'issues': issues,
                    'pulls': pulls,
                    'size_kb': size,
                    'file_count': file_count
                }
                # Check for relevance
                if any(kw.lower() in (repo.description or '').lower() or kw.lower() in (repo.language or '').lower() or kw.lower() in ' '.join(repo_data['topics']).lower() for kw in job_keywords):
                    relevant_repos.append(repo_data)
            return {
                'total_repos': repos.totalCount,
                'relevant_repos': relevant_repos,
                'github_url': f'https://github.com/{github_username}'
            }
        except Exception as e:
            return {'error': str(e)}
    def __init__(self, api_key: str = None):
        """
        Initialize the Resume Parser Agent with Gemini API
        
        Args:
            api_key: Gemini API key. If None, will look for GEMINI_API_KEY in environment
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Gemini API key is required. Set GEMINI_API_KEY environment variable or pass api_key parameter")
        
        # Configure Gemini
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text from PDF resume
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text from the PDF
        """
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            raise Exception(f"Error reading PDF file: {str(e)}")
        
        return text
    
    def clean_resume_text(self, text: str) -> str:
        """
        Clean and preprocess resume text
        
        Args:
            text: Raw text from resume
            
        Returns:
            Cleaned resume text
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep relevant ones
        text = re.sub(r'[^\w\s.,;:!?@#%&*+/-]', '', text)
        return text.strip()
    
    def parse_with_gemini(self, job_description: str, resume_text: str) -> Dict:
        """
        Use Gemini to analyze resume against job description
        
        Args:
            job_description: The job description text
            resume_text: The extracted resume text
            
        Returns:
            Analysis results as a dictionary
        """
        prompt = f"""
        Analyze this resume against the given job description and provide a comprehensive evaluation.
        
        JOB DESCRIPTION:
        {job_description}
        
        RESUME:
        {resume_text}
        
        Please provide a JSON response with the following structure:
        {{
            "suitability_score": 0-100,
            "skills_match": {{
                "matched_skills": [],
                "missing_skills": []
            }},
            "experience_evaluation": {{
                "years_experience": 0,
                "relevance": "low/medium/high"
            }},
            "education_evaluation": {{
                "degree_match": true/false,
                "education_level": "description of education level"
            }},
            "keyword_analysis": {{
                "matched_keywords": [],
                "missing_keywords": []
            }},
            "strengths": [],
            "weaknesses": [],
            "overall_assessment": "text summary"
        }}
        
        Be objective and focus on factual matches between the resume and job requirements.
        """
        
        try:
            response = self.model.generate_content(prompt)
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                # Fallback if JSON not found
                return {
                    "suitability_score": 0,
                    "skills_match": {"matched_skills": [], "missing_skills": []},
                    "experience_evaluation": {"years_experience": 0, "relevance": "low"},
                    "education_evaluation": {"degree_match": False, "education_level": ""},
                    "keyword_analysis": {"matched_keywords": [], "missing_keywords": []},
                    "strengths": [],
                    "weaknesses": [],
                    "overall_assessment": "Failed to parse response"
                }
        except Exception as e:
            raise Exception(f"Error calling Gemini API: {str(e)}")
    
    def analyze_resume(self, job_description: str, pdf_path: str) -> Dict:
        """
        Main method to analyze a resume against a job description
        
        Args:
            job_description: The job description text
            pdf_path: Path to the resume PDF file
            
        Returns:
            Comprehensive analysis results
        """
        # Extract text from PDF
        resume_text = self.extract_text_from_pdf(pdf_path)
        cleaned_text = self.clean_resume_text(resume_text)

        # Analyze with Gemini (resume only)
        analysis = self.parse_with_gemini(job_description, cleaned_text)

        # GitHub analysis
        github_username = self.extract_github_profile(resume_text)
        github_analysis = None
        if github_username:
            job_keywords = job_description.split()  # Simple keyword extraction
            github_analysis = self.analyze_github_projects(github_username, job_keywords)

        # Combined Gemini scoring
        github_summary = self.summarize_github_for_gemini(github_analysis)
        combined_score = self.combined_gemini_score(job_description, cleaned_text, github_summary)

        return {
            'resume_analysis': analysis,
            'github_analysis': github_analysis,
            'combined_score': combined_score
        }
    def save_report_pdf(self, report_text: str, pdf_path: str):
        """Save the report as a formatted PDF using reportlab."""
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
        from reportlab.lib.utils import simpleSplit
        c = canvas.Canvas(pdf_path, pagesize=letter)
        width, height = letter
        margin = 40
        y = height - margin
        lines = simpleSplit(report_text, 'Helvetica', 10, width - 2 * margin)
        c.setFont('Helvetica', 10)
        for line in lines:
            if y < margin:
                c.showPage()
                y = height - margin
                c.setFont('Helvetica', 10)
            c.drawString(margin, y, line)
            y -= 14
        c.save()
    
    def format_output(self, analysis: Dict) -> str:
        """
        Format the analysis results for display
        
        Args:
            analysis: The analysis results dictionary
            
        Returns:
            Formatted string output
        """
        # If the analysis dict contains both resume_analysis and github_analysis, format accordingly
        if 'resume_analysis' in analysis:
            resume_analysis = analysis['resume_analysis']
            github_analysis = analysis.get('github_analysis')
            output = f"=== RESUME ANALYSIS REPORT ===\n\n"
            output += f"Suitability Score: {resume_analysis.get('suitability_score', 0)}/100\n\n"
            # Skills match
            skills = resume_analysis.get('skills_match', {})
            output += "SKILLS ANALYSIS:\n"
            output += f"Matched Skills: {', '.join(skills.get('matched_skills', [])) or 'None'}\n"
            output += f"Missing Skills: {', '.join(skills.get('missing_skills', [])) or 'None'}\n\n"
            # Experience
            experience = resume_analysis.get('experience_evaluation', {})
            output += "EXPERIENCE EVALUATION:\n"
            output += f"Years of Experience: {experience.get('years_experience', 0)}\n"
            output += f"Relevance: {experience.get('relevance', 'Unknown')}\n\n"
            # Education
            education = resume_analysis.get('education_evaluation', {})
            output += "EDUCATION EVALUATION:\n"
            output += f"Degree Match: {'Yes' if education.get('degree_match') else 'No'}\n"
            output += f"Education Level: {education.get('education_level', 'Not specified')}\n\n"
            # Keywords
            keywords = resume_analysis.get('keyword_analysis', {})
            output += "KEYWORD ANALYSIS:\n"
            output += f"Matched Keywords: {', '.join(keywords.get('matched_keywords', [])) or 'None'}\n"
            output += f"Missing Keywords: {', '.join(keywords.get('missing_keywords', [])) or 'None'}\n\n"
            # Strengths and weaknesses
            output += "STRENGTHS:\n"
            for strength in resume_analysis.get('strengths', []):
                output += f"- {strength}\n"
            output += "\nWEAKNESSES:\n"
            for weakness in resume_analysis.get('weaknesses', []):
                output += f"- {weakness}\n"
            output += f"\nOVERALL ASSESSMENT:\n{resume_analysis.get('overall_assessment', 'No assessment available')}\n\n"
            # GitHub section
            output += "=== GITHUB PROJECTS ANALYSIS ===\n"
            if github_analysis:
                if 'error' in github_analysis:
                    output += f"Error fetching GitHub data: {github_analysis['error']}\n"
                else:
                    output += f"GitHub Profile: {github_analysis.get('github_url', 'N/A')}\n"
                    output += f"Total Public Repos: {github_analysis.get('total_repos', 'N/A')}\n"
                    output += "Relevant Projects:\n"
                    for repo in github_analysis.get('relevant_repos', []):
                        output += f"- {repo['name']} ({repo['language']})\n  {repo['description']}\n  Topics: {', '.join(repo['topics'])}\n  Languages: {', '.join(repo['languages'])}\n  Created At: {repo['created_at']} | Last Updated: {repo['updated_at']}\n  URL: {repo['url']}\n  Stars: {repo['stars']} | Forks: {repo['forks']} | Last Commit: {repo['last_commit']}\n  Contributors: {', '.join(repo['contributors'])}\n  Issues: {repo['issues']} | Pull Requests: {repo['pulls']}\n  Repo Size: {repo['size_kb']} KB | File Count: {repo['file_count']}\n  README (first 300 chars): {repo['readme'][:300].replace(chr(10), ' ')}\n"
                    if not github_analysis.get('relevant_repos'):
                        output += "No relevant projects found.\n"
            else:
                output += "No GitHub profile found in resume.\n"
            return output
        else:
            # Fallback to old format if only resume analysis is present
            output = f"=== RESUME ANALYSIS REPORT ===\n\n"
            output += f"Suitability Score: {analysis.get('suitability_score', 0)}/100\n\n"
            # Skills match
            skills = analysis.get('skills_match', {})
            output += "SKILLS ANALYSIS:\n"
            output += f"Matched Skills: {', '.join(skills.get('matched_skills', [])) or 'None'}\n"
            output += f"Missing Skills: {', '.join(skills.get('missing_skills', [])) or 'None'}\n\n"
            # Experience
            experience = analysis.get('experience_evaluation', {})
            output += "EXPERIENCE EVALUATION:\n"
            output += f"Years of Experience: {experience.get('years_experience', 0)}\n"
            output += f"Relevance: {experience.get('relevance', 'Unknown')}\n\n"
            # Education
            education = analysis.get('education_evaluation', {})
            output += "EDUCATION EVALUATION:\n"
            output += f"Degree Match: {'Yes' if education.get('degree_match') else 'No'}\n"
            output += f"Education Level: {education.get('education_level', 'Not specified')}\n\n"
            # Keywords
            keywords = analysis.get('keyword_analysis', {})
            output += "KEYWORD ANALYSIS:\n"
            output += f"Matched Keywords: {', '.join(keywords.get('matched_keywords', [])) or 'None'}\n"
            output += f"Missing Keywords: {', '.join(keywords.get('missing_keywords', [])) or 'None'}\n\n"
            # Strengths and weaknesses
            output += "STRENGTHS:\n"
            for strength in analysis.get('strengths', []):
                output += f"- {strength}\n"
            output += "\nWEAKNESSES:\n"
            for weakness in analysis.get('weaknesses', []):
                output += f"- {weakness}\n"
            output += f"\nOVERALL ASSESSMENT:\n{analysis.get('overall_assessment', 'No assessment available')}"
            return output

# Example usage and integration point
if __name__ == "__main__":
    # Initialize the parser
    parser = ResumeParserAgent()
    
    # Example job description
    job_description = """
    We are seeking a motivated Junior Machine Learning Engineer with:
    -A Bachelor's degree in Computer Science, Data Science, or a related field
    -Strong foundational knowledge of Machine Learning algorithms and principles
    -Proficiency in Python and key ML libraries (e.g., Scikit-learn, TensorFlow, or PyTorch)
    -Experience with data preprocessing, visualization, and analysis (Pandas, NumPy, Matplotlib)
    -Familiarity with version control systems, preferably Git
    -Strong problem-solving abilities and a passion for innovation
    -Excellent communication skills and ability to work in a collaborative team environment
    """
    
    # Analyze a resume
    try:
        analysis = parser.analyze_resume(job_description, r"C:\Users\thend\Downloads\Resumes_CVs\Thendral_Kabilan_Resume.pdf")
        report = parser.format_output(analysis)
        print(report)

        # Save results for other agents (integration point)
        with open("resume_analysis.json", "w") as f:
            json.dump(analysis, f, indent=2)

        # Save the report as a PDF
        pdf_path = "resume_analysis_report.pdf"
        parser.save_report_pdf(report, pdf_path)
        print(f"PDF report saved to {pdf_path}")

    except Exception as e:
        print(f"Error analyzing resume: {str(e)}")