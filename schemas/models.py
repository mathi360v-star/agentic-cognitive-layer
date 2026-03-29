from typing import TypedDict, List, Optional, Dict, Any

class AgenticState(TypedDict):
    """The Universal Data Bus for the Swarm Architecture."""
    current_topic: str           
    domain: str                  
    target_language: str
    problem_statement: str
    difficulty_level: str
    
    problem_is_valid: bool       
    audit_feedback: str          
    
    fundamental_laws: str        
    difficulty_tier: str         
    final_correct_code: str      
    
    proposed_code: str 
    execution_success: bool          
    
    rca_history: List[Dict[str, Any]]      
    iteration_count: int
    total_graph_steps: int # CRITICAL: The Global Circuit Breaker