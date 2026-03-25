import Anthropic from '@anthropic-ai/sdk';

export class SqlGenerator {
  private client: Anthropic;
  private fullyQualifiedTable: string;
  private schemaDescription: string;

  constructor(apiKey: string, projectId: string, dataset: string, table: string, schemaDescription?: string) {
    this.client = new Anthropic({ apiKey });
    this.fullyQualifiedTable = `\`${projectId}.${dataset}.${table}\``;
    this.schemaDescription = schemaDescription || '';
  }

  async generate(naturalLanguageQuery: string): Promise<string> {
    const systemPrompt = `You are a SQL generator for Google BigQuery. You MUST return ONLY a valid BigQuery Standard SQL SELECT statement — no explanation, no markdown, no code fences, just raw SQL.

TABLE: ${this.fullyQualifiedTable}

${this.schemaDescription}

RULES:
1. ONLY generate SELECT statements. Never generate INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, MERGE, TRUNCATE, or any mutation.
2. Always query from ${this.fullyQualifiedTable}.
3. Use BigQuery Standard SQL syntax.
4. Use LIKE with wildcards for partial text matching. Be case-insensitive by using LOWER() when doing text matching.
5. Do NOT add a LIMIT clause unless the user explicitly asks for a specific number of results.
6. If the user asks for a count or aggregate, use COUNT(*), GROUP BY, etc.
7. Return ONLY the SQL query text with no surrounding formatting.`;

    const response = await this.client.messages.create({
      model: 'claude-sonnet-4-6',
      max_tokens: 1024,
      messages: [{ role: 'user', content: naturalLanguageQuery }],
      system: systemPrompt,
    });

    const textBlock = response.content.find((block) => block.type === 'text');
    if (!textBlock || textBlock.type !== 'text') {
      throw new Error('Claude returned no text content');
    }

    let sql = textBlock.text.trim();
    sql = sql.replace(/^```(?:sql)?\s*/i, '').replace(/\s*```$/i, '').trim();

    const norm = sql.toUpperCase().replace(/\s+/g, ' ').trim();
    if (!norm.startsWith('SELECT') && !norm.startsWith('WITH')) {
      throw new Error('Generated SQL is not a SELECT query.');
    }
    const forbidden = ['INSERT', 'UPDATE', 'DELETE', 'DROP', 'ALTER', 'CREATE', 'TRUNCATE', 'MERGE', 'GRANT', 'REVOKE'];
    for (const kw of forbidden) {
      if (new RegExp(`\\b${kw}\\b`, 'i').test(sql)) {
        throw new Error(`Generated SQL contains forbidden keyword "${kw}".`);
      }
    }
    return sql;
  }
}
