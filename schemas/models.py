from pydantic import BaseModel, Field
from typing import TypedDict, List, Optional, Dict, Any

class ProblemAudit(BaseModel):
    is_valid: bool = Field(..., description="True if mathematically/physically possible. False if flawed or paradoxical.")
    flaw_reasoning: str = Field(..., description="Explanation of why it failed, or 'Valid'.")

class CurriculumOutput(BaseModel):
    domain: str = Field(..., description="STEM sector, e.g., 'Aerospace', 'Calculus', 'C++ Memory', 'Quantum Physics'.")
    target_language: str = Field(..., description="Programming language ('python', 'c') OR 'Agnostic/Math' for theoretical physics/math proofs.")
    problem_statement: str = Field(..., description="Detailed engineering, math, or physics problem.")
    difficulty_level: str = Field(..., description="Complexity: 'Moderate', 'Advanced', 'Elite'.")
    hidden_unit_tests: str = Field(..., description="Mathematical proofs, physical boundary conditions, or logic tests.")

class AgenticState(TypedDict):
    """The Universal Data Bus for the Swarm Architecture."""
    current_topic: str           
    domain: str                  
    target_language: str
    problem_statement: str
    difficulty_level: str
    hidden_unit_tests: str 
    
    problem_is_valid: bool       
    audit_feedback: str    
    
    # --- NEW ENTERPRISE FIELDS FOR FORMAL LOGIC & ENTROPY ---
    fundamental_laws: Optional[str]  
    difficulty_tier: Optional[str]   
    final_correct_code: Optional[str]
    # --------------------------------------------------------
    
    proposed_code: Optional[str] 
    red_team_critique: Optional[str] 
    execution_success: bool          
    traceback: Optional[str]        
    
    rca_history: List[Dict[str, Any]]      
    iteration_count: int