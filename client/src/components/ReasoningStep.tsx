import { useState } from 'react';
import type { Step } from '../types';

interface ReasoningStepProps {
  step: Step;
}

// Matches the "N. document: X | section: Y | score: Z\n   excerpt: ..." blocks
// emitted by search_knowledge_base._format_results (also embedded twice inside
// compare_sections' "=== Results from ... ===" output) — see
// server/app/agent/tools/search_knowledge_base.py.
const SEARCH_RESULT_ENTRY_RE = /^\d+\.\s*document:/gm;
const SEARCH_RESULT_BLOCK_RE =
  /\d+\.\s*document:\s*(.+?)\s*\|\s*section:\s*(.+?)\s*\|\s*score:\s*(.+?)\s*\n\s*excerpt:\s*([\s\S]*?)(?=\n\d+\.\s*document:|\n===|$)/g;

function truncate(text: string, maxLength: number): string {
  const clean = text.trim();
  if (clean.length <= maxLength) return clean;
  const sliced = clean.slice(0, maxLength);
  const lastSpace = sliced.lastIndexOf(' ');
  const base = lastSpace > maxLength * 0.5 ? sliced.slice(0, lastSpace) : sliced;
  return `${base.trimEnd()}...`;
}

function categoryLabel(category: string): string {
  return category.toLowerCase() === 'hr' ? 'HR' : category.charAt(0).toUpperCase() + category.slice(1);
}

function formatParamValue(value: unknown): string {
  if (typeof value === 'string') return value;
  return JSON.stringify(value);
}

function getActionLabel(tool: string | null | undefined): string {
  switch (tool) {
    case 'search_knowledge_base':
      return 'Search';
    case 'get_document':
      return 'Read Document';
    case 'list_documents':
      return 'List Documents';
    case 'compare_sections':
      return 'Compare';
    case 'calculate_or_verify':
      return 'Calculate';
    default:
      return 'Action';
  }
}

function getLabel(step: Step): string {
  switch (step.type) {
    case 'thought':
      return 'Thinking';
    case 'observation':
      return 'Result';
    case 'retry':
      return 'Retry';
    case 'action':
      return getActionLabel(step.tool);
    default:
      return 'Step';
  }
}

function summarizeAction(tool: string | null | undefined, params: Record<string, unknown> | null | undefined): string {
  const p = params ?? {};
  switch (tool) {
    case 'search_knowledge_base': {
      const query = typeof p.query === 'string' ? p.query : '';
      const category =
        typeof p.category === 'string' && p.category ? ` within ${categoryLabel(p.category)} documents` : '';
      return `Searching the knowledge base for "${query}"${category}`;
    }
    case 'get_document': {
      const name = typeof p.document_name === 'string' ? p.document_name : 'the document';
      return `Reading the full document: ${name}`;
    }
    case 'list_documents':
      return 'Checking which documents are available';
    case 'compare_sections': {
      const doc1 = typeof p.doc1 === 'string' ? p.doc1 : 'the first document';
      const doc2 = typeof p.doc2 === 'string' ? p.doc2 : 'the second document';
      const topic = typeof p.topic === 'string' ? p.topic : '';
      return `Comparing ${doc1} and ${doc2} on "${topic}"`;
    }
    case 'calculate_or_verify': {
      const expression = typeof p.expression === 'string' ? p.expression : '';
      return `Calculating: ${expression}`;
    }
    default:
      return tool ? `Running ${tool}` : 'Running a tool';
  }
}

function summarizeObservation(content: string): string {
  const trimmed = content.trim();
  const matches = trimmed.match(SEARCH_RESULT_ENTRY_RE);
  const count = matches ? matches.length : 0;
  if (count > 0) {
    return `Found ${count} result${count === 1 ? '' : 's'}`;
  }
  if (trimmed.includes('No relevant chunks found.')) {
    return 'No relevant results found';
  }
  return truncate(trimmed, 80);
}

function summarizeRetry(content: string): string {
  const lower = content.toLowerCase();
  const reason = lower.includes('malformed') ? 'malformed output' : 'empty response';
  return `Retrying — ${reason}`;
}

function getSummary(step: Step): string {
  switch (step.type) {
    case 'thought':
      return truncate(step.content ?? '', 100);
    case 'retry':
      return summarizeRetry(step.content ?? '');
    case 'observation':
      return summarizeObservation(step.content ?? '');
    case 'action':
      return summarizeAction(step.tool, step.params);
    default:
      return truncate(step.content ?? '', 100);
  }
}

function formatObservationDetail(content: string): string {
  const trimmed = content.trim();
  if (!trimmed) return '(empty)';
  if (!/^\d+\.\s*document:/m.test(trimmed)) {
    return trimmed;
  }
  return trimmed
    .replace(
      SEARCH_RESULT_BLOCK_RE,
      (_match, doc: string, section: string, score: string, excerpt: string) =>
        `${doc.trim()} · ${section.trim()} · score ${score.trim()}\n${excerpt.trim()}`
    )
    .trim();
}

function getDetail(step: Step): string {
  switch (step.type) {
    case 'action': {
      const header = `Tool: ${step.tool ?? 'unknown'}`;
      const paramLines = Object.entries(step.params ?? {}).map(
        ([key, value]) => `${key}: ${formatParamValue(value)}`
      );
      return paramLines.length > 0 ? `${header}\n${paramLines.join('\n')}` : `${header}\n(no parameters)`;
    }
    case 'observation':
      return formatObservationDetail(step.content ?? '');
    case 'thought':
    case 'retry':
    default:
      return (step.content ?? '').trim() || '(empty)';
  }
}

function ReasoningStep({ step }: ReasoningStepProps) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div className="reasoning-step">
      <button
        type="button"
        className="reasoning-step-toggle"
        onClick={() => setExpanded((prev) => !prev)}
        aria-expanded={expanded}
      >
        <span className="reasoning-step-header">
          <span className="reasoning-step-label">{getLabel(step)}</span>
          <span className="reasoning-step-chevron">{expanded ? '▾' : '▸'}</span>
        </span>
        <span className="reasoning-step-summary">{getSummary(step)}</span>
      </button>
      {expanded && (
        <div className="reasoning-step-body">
          <div className="reasoning-step-detail">{getDetail(step)}</div>
        </div>
      )}
    </div>
  );
}

export default ReasoningStep;
