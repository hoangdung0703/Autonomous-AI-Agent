import axios from 'axios';
import type { ChatResponse, HealthResponse } from '../types';

const apiClient = axios.create({
  baseURL: import.meta.env.VITE_API_URL,
});

// User-facing text for a failed /api/chat call — deliberately generic so it
// never leaks a raw axios/HTTP string (e.g. "Request failed with status
// code 500") into the chat UI. The real error is still logged to the
// console for debugging.
const CHAT_FAILURE_MESSAGE = 'Something went wrong while the agent was thinking. Please try again.';

export async function postChat(question: string, conversationId?: string): Promise<ChatResponse> {
  try {
    const response = await apiClient.post<ChatResponse>('/api/chat', {
      question,
      conversation_id: conversationId ?? null,
    });
    return response.data;
  } catch (error) {
    console.error('postChat failed:', error);
    throw new Error(CHAT_FAILURE_MESSAGE);
  }
}

export async function getHealth(): Promise<HealthResponse> {
  const response = await apiClient.get<HealthResponse>('/api/health');
  return response.data;
}
