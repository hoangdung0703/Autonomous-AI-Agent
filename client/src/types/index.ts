// TypeScript types mirroring server/app/models.py exactly (Section 8 of requirements.md).

export type StepType = 'thought' | 'action' | 'observation' | 'retry';

export interface Step {
  type: StepType;
  content?: string | null;
  tool?: string | null;
  params?: Record<string, unknown> | null;
}

export interface Source {
  document_name: string;
  section_title: string;
  excerpt: string;
}

export interface ChatRequest {
  question: string;
  conversation_id?: string | null;
}

export interface ChatResponse {
  conversation_id: string;
  steps: Step[];
  answer: string;
  sources: Source[];
}

export interface HealthResponse {
  status: string;
  qdrant: string;
  supabase: string;
  gemini: string;
  documents_indexed: number;
}
