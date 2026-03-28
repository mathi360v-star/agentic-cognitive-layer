from pydantic import BaseModel, Field
from typing import TypedDict, List, Optional, Dict, Any

# --------------------------------------------------------------
# Pydantic Models for Structured LLM Outputs
# --------------------------------------------------------------

class ProblemAudit(BaseModel):
    """Used by the Verifier to output structured logic checks."""
    is_valid: bool = Field(..., description="True if mathematically/physically possible. False if flawed or paradoxical.")
    flaw_reasoning: str = Field(..., description="Explanation of why it failed, or 'Valid'.")

class CurriculumOutput(BaseModel):
    """Used by the Professor to output multi-domain STEM problems."""
    domain: str = Field(..., description="STEM sector, e.g., 'Aerospace', 'Calculus', 'C++ Memory'.")
    target_language: str = Field(..., description="Programming language ('python', 'c') OR 'Agnostic/Math'.")
    problem_statement: str = Field(..., description="Detailed engineering, math, or physics problem.")
    difficulty_level: str = Field(..., description="Complexity: 'Moderate', 'Advanced', 'Elite'.")
    hidden_unit_tests: str = Field(..., description="Mathematical proofs, physical boundary conditions, or logic tests.")

# --------------------------------------------------------------
# The Universal Data Bus (LangGraph State)
# --------------------------------------------------------------

class AgenticState(TypedDict):
    """
    The Universal Data Bus for the Swarm Architecture.
    All agents read and write to this dictionary.
    """
    current_topic: str           
    domain: str                  
    target_language: str
    problem_statement: str
    difficulty_level: str
    hidden_unit_tests: str 
    
    # Logic Gates
    problem_is_valid: bool       # For Epistemic Shield & Verifier
    audit_feedback: str          # For Judge -> Analyst -> Scientist feedback loop
    
    # New V3 Enterprise Fields
    fundamental_laws: Optional[str]  # Locked laws by the Physicist
    difficulty_tier: Optional[str]   # Dynamically assigned by iteration count
    final_correct_code: Optional[str]# Final gold solution for harvesting
    
    # Execution & Reasoning
    proposed_code: Optional[str] 
    execution_success: bool          
    traceback: Optional[str]        
    
    # History & Metadata
    rca_history: List[Dict[str, Any]]      
    iteration_count: int