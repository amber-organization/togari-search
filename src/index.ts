#!/usr/bin/env node

/**
 * togari-amber — MCP Server for the Amber Health Network Platform
 *
 * Provides development, deployment, and operations tools for the Amber iOS app
 * (SwiftUI) with Fastify backend on GCP (Cloud Run + PostgreSQL + Privy auth).
 *
 * Transport modes:
 *   - stdio mode for Claude Desktop (default)
 *   - HTTP/SSE mode when PORT is set (Railway deployment)
 */

import { McpServer } from '@modelcontextprotocol/sdk/server/mcp';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio';
import { SSEServerTransport } from '@modelcontextprotocol/sdk/server/sse';
import express from 'express';
import { z } from 'zod';

const AMBER_API_URL = process.env.AMBER_API_URL || 'http://localhost:3000';
const AMBER_REPO_PATH = process.env.AMBER_REPO_PATH || '';

// ---------------------------------------------------------------------------
// Tool Registration
// ---------------------------------------------------------------------------

function registerTools(server: McpServer): void {
  // =========================================================================
  // 1. HEALTH CHECK
  // =========================================================================

  server.tool(
    'amber_health_check',
    'Check Amber backend health by hitting the Fastify health endpoint. ' +
      'Returns status, version, and uptime information.',
    {},
    async () => {
      try {
        const url = `${AMBER_API_URL}/health`;
        const resp = await fetch(url, { signal: AbortSignal.timeout(10000) });
        const data = await resp.json();
        return {
          content: [{
            type: 'text' as const,
            text: JSON.stringify({
              status: resp.ok ? 'healthy' : 'unhealthy',
              http_status: resp.status,
              url,
              response: data,
            }, null, 2),
          }],
        };
      } catch (error) {
        const msg = error instanceof Error ? error.message : String(error);
        return {
          content: [{
            type: 'text' as const,
            text: JSON.stringify({
              status: 'unreachable',
              url: `${AMBER_API_URL}/health`,
              error: msg,
            }, null, 2),
          }],
          isError: true,
        };
      }
    }
  );

  // =========================================================================
  // 2. LIST SPRINT TASKS (Prompt Delegation to Linear)
  // =========================================================================

  server.tool(
    'amber_list_sprint_tasks',
    'Returns a formatted summary of all Amber Sprint 1 MVP tasks from Linear. ' +
      'Uses prompt delegation — returns instructions for Claude to use Linear MCP tools.',
    {
      sprint: z.string().default('Sprint 1').describe('Sprint name to filter by'),
      status_filter: z.enum(['all', 'todo', 'in_progress', 'done', 'backlog']).default('all').describe('Filter tasks by status'),
    },
    async ({ sprint, status_filter }) => {
      const delegation = {
        action: 'prompt_delegation',
        instructions: `Use the Linear MCP tools to list all issues for the Amber project. Follow these steps:
1. Use list_teams to find the ENGBMA team
2. Use list_issues with the team ID to get all issues
3. Filter by sprint/cycle name containing "${sprint}"
${status_filter !== 'all' ? `4. Filter by status: ${status_filter}` : '4. Show all statuses'}
5. Format the results as a table with columns: ID, Title, Status, Assignee, Priority
6. Group by status (Backlog, Todo, In Progress, Done)
7. Include a summary count at the top`,
        context: {
          project: 'Amber',
          team_prefix: 'ENGBMA',
          sprint,
          status_filter,
        },
      };
      return {
        content: [{
          type: 'text' as const,
          text: JSON.stringify(delegation, null, 2),
        }],
      };
    }
  );

  // =========================================================================
  // 3. GET TASK STATUS (Prompt Delegation to Linear)
  // =========================================================================

  server.tool(
    'amber_get_task_status',
    'Get the status of a specific ENGBMA issue from Linear. ' +
      'Uses prompt delegation — returns instructions for Claude to use Linear MCP tools.',
    {
      issue_id: z.string().describe('Linear issue identifier (e.g., "ENGBMA-42")'),
    },
    async ({ issue_id }) => {
      const delegation = {
        action: 'prompt_delegation',
        instructions: `Use the Linear MCP tools to get details for issue ${issue_id}:
1. Use get_issue with identifier "${issue_id}"
2. Return: title, description, status, assignee, priority, labels, created date, updated date
3. If the issue has sub-issues, list them too
4. Include any comments on the issue`,
        context: {
          issue_id,
          team_prefix: 'ENGBMA',
        },
      };
      return {
        content: [{
          type: 'text' as const,
          text: JSON.stringify(delegation, null, 2),
        }],
      };
    }
  );

  // =========================================================================
  // 4. UPDATE TASK (Prompt Delegation to Linear)
  // =========================================================================

  server.tool(
    'amber_update_task',
    'Update a Linear task status, assignee, or other fields. ' +
      'Uses prompt delegation — returns instructions for Claude to use Linear MCP tools.',
    {
      issue_id: z.string().describe('Linear issue identifier (e.g., "ENGBMA-42")'),
      status: z.string().optional().describe('New status (e.g., "In Progress", "Done", "Todo")'),
      assignee: z.string().optional().describe('Assignee name or email'),
      priority: z.number().min(0).max(4).optional().describe('Priority: 0=No priority, 1=Urgent, 2=High, 3=Medium, 4=Low'),
      comment: z.string().optional().describe('Comment to add to the issue'),
    },
    async ({ issue_id, status, assignee, priority, comment }) => {
      const updates: Record<string, unknown> = {};
      if (status) updates.status = status;
      if (assignee) updates.assignee = assignee;
      if (priority !== undefined) updates.priority = priority;

      const delegation = {
        action: 'prompt_delegation',
        instructions: `Use the Linear MCP tools to update issue ${issue_id}:
1. Use save_issue to update the issue with these changes: ${JSON.stringify(updates)}
${comment ? `2. Use save_comment to add this comment: "${comment}"` : ''}
3. After updating, fetch and return the updated issue details`,
        context: {
          issue_id,
          updates,
          comment,
        },
      };
      return {
        content: [{
          type: 'text' as const,
          text: JSON.stringify(delegation, null, 2),
        }],
      };
    }
  );

  // =========================================================================
  // 5. LIST API ROUTES (Prompt Delegation to filesystem)
  // =========================================================================

  server.tool(
    'amber_list_api_routes',
    'List all Fastify API routes defined in the Amber backend. ' +
      'Returns instructions for Claude to parse route files from the repo.',
    {
      filter: z.string().optional().describe('Filter routes by path pattern (e.g., "auth", "user", "health")'),
    },
    async ({ filter }) => {
      const repoPath = AMBER_REPO_PATH;
      const delegation = {
        action: 'prompt_delegation',
        instructions: `Use filesystem tools to list all Fastify API routes in the Amber backend:
1. Search for route definition files in the backend directory: ${repoPath ? repoPath + '/backend' : 'the amber repo backend/'}
2. Look for patterns like: fastify.get, fastify.post, fastify.put, fastify.delete, fastify.register
3. Also check for route files in src/routes/ or src/api/ directories
4. Parse each route file and extract: HTTP method, path, handler name, any auth requirements
${filter ? `5. Filter results to routes containing "${filter}"` : '5. Show all routes'}
6. Format as a table: Method | Path | Auth Required | Handler`,
        context: {
          repo_path: repoPath,
          filter: filter || null,
          common_locations: [
            'backend/src/routes/',
            'backend/src/api/',
            'backend/src/index.ts',
            'backend/src/app.ts',
          ],
        },
      };
      return {
        content: [{
          type: 'text' as const,
          text: JSON.stringify(delegation, null, 2),
        }],
      };
    }
  );

  // =========================================================================
  // 6. DB SCHEMA (Prompt Delegation to filesystem)
  // =========================================================================

  server.tool(
    'amber_db_schema',
    'Show the current Drizzle database schema definitions for the Amber backend. ' +
      'Returns instructions for Claude to read and display schema files.',
    {
      table: z.string().optional().describe('Specific table name to show (e.g., "users", "connections")'),
    },
    async ({ table }) => {
      const repoPath = AMBER_REPO_PATH;
      const delegation = {
        action: 'prompt_delegation',
        instructions: `Use filesystem tools to read and display the Drizzle database schema:
1. Find schema files in: ${repoPath ? repoPath + '/backend' : 'the amber repo backend/'}
2. Look in common locations: src/db/schema/, src/schema/, drizzle/, src/db/schema.ts
3. Read all schema definition files (*.ts files with pgTable, pgEnum, etc.)
${table ? `4. Focus on the "${table}" table definition` : '4. Show all table definitions'}
5. For each table, list: table name, columns (name, type, constraints), indexes, relations
6. If there are Drizzle relations defined, show those too
7. Format clearly with column types and constraints`,
        context: {
          repo_path: repoPath,
          table: table || null,
          common_locations: [
            'backend/src/db/schema/',
            'backend/src/db/schema.ts',
            'backend/src/schema/',
            'backend/drizzle/',
          ],
        },
      };
      return {
        content: [{
          type: 'text' as const,
          text: JSON.stringify(delegation, null, 2),
        }],
      };
    }
  );

  // =========================================================================
  // 7. DEPLOY STATUS (Prompt Delegation)
  // =========================================================================

  server.tool(
    'amber_deploy_status',
    'Check deployment status of the Amber backend on GCP Cloud Run or Railway. ' +
      'Returns instructions for Claude to check deployment status via CLI tools.',
    {
      platform: z.enum(['cloud_run', 'railway', 'both']).default('both').describe('Which platform to check'),
    },
    async ({ platform }) => {
      const delegation = {
        action: 'prompt_delegation',
        instructions: `Check Amber deployment status:
${platform === 'cloud_run' || platform === 'both' ? `
**GCP Cloud Run:**
1. Run: gcloud run services describe amber-backend --region=us-central1 --format=json
2. Extract: service URL, latest revision, traffic split, last deployed timestamp
3. Run: gcloud run revisions list --service=amber-backend --region=us-central1 --limit=5
4. Show recent revision history` : ''}
${platform === 'railway' || platform === 'both' ? `
**Railway:**
1. Run: railway status (if Railway CLI is available)
2. Or check the Railway dashboard URL
3. Show: deployment status, environment, last deploy time` : ''}
4. Also check if the health endpoint responds: curl ${AMBER_API_URL}/health`,
        context: {
          platform,
          api_url: AMBER_API_URL,
        },
      };
      return {
        content: [{
          type: 'text' as const,
          text: JSON.stringify(delegation, null, 2),
        }],
      };
    }
  );

  // =========================================================================
  // 8. CREATE BRANCH (Prompt Delegation)
  // =========================================================================

  server.tool(
    'amber_create_branch',
    'Create a feature branch from an ENGBMA Linear issue. ' +
      'Uses prompt delegation — tells Claude to create a git branch following naming conventions.',
    {
      issue_id: z.string().describe('Linear issue identifier (e.g., "ENGBMA-42")'),
      branch_type: z.enum(['feature', 'fix', 'chore', 'refactor']).default('feature').describe('Branch type prefix'),
    },
    async ({ issue_id, branch_type }) => {
      const delegation = {
        action: 'prompt_delegation',
        instructions: `Create a feature branch for ${issue_id}:
1. First, use the Linear MCP tools to get the issue title for ${issue_id}
2. Generate a branch name: ${branch_type}/${issue_id.toLowerCase()}-<slugified-title>
   Example: feature/engbma-42-add-user-authentication
3. In the Amber repo directory, run:
   git checkout main && git pull origin main
   git checkout -b <branch-name>
4. Confirm the branch was created successfully
5. Return the branch name`,
        context: {
          issue_id,
          branch_type,
          repo_path: AMBER_REPO_PATH,
        },
      };
      return {
        content: [{
          type: 'text' as const,
          text: JSON.stringify(delegation, null, 2),
        }],
      };
    }
  );

  // =========================================================================
  // 9. CREATE PR (Prompt Delegation)
  // =========================================================================

  server.tool(
    'amber_create_pr',
    'Create a pull request for review with Karthik as reviewer. ' +
      'Uses prompt delegation — tells Claude to create a PR via gh CLI.',
    {
      title: z.string().describe('PR title'),
      description: z.string().optional().describe('PR description/body'),
      issue_id: z.string().optional().describe('Linear issue ID to link (e.g., "ENGBMA-42")'),
      draft: z.boolean().default(false).describe('Create as draft PR'),
    },
    async ({ title, description, issue_id, draft }) => {
      const body = [
        description || '',
        '',
        issue_id ? `Resolves: ${issue_id}` : '',
        '',
        '## Checklist',
        '- [ ] TypeScript compiles without errors',
        '- [ ] API routes tested',
        '- [ ] Database migrations reviewed',
        '- [ ] iOS changes tested in simulator',
      ].filter(Boolean).join('\n');

      const delegation = {
        action: 'prompt_delegation',
        instructions: `Create a pull request for the Amber repo:
1. Make sure all changes are committed and pushed to the current branch
2. Run: gh pr create --repo amber-organization/amber --title "${title}" --body "<body>" ${draft ? '--draft' : ''} --reviewer karthik
3. If the PR was created successfully, return the PR URL
4. ${issue_id ? `Link this PR to Linear issue ${issue_id}` : 'No Linear issue to link'}`,
        context: {
          title,
          body,
          issue_id,
          draft,
          repo: 'amber-organization/amber',
          reviewer: 'karthik',
        },
      };
      return {
        content: [{
          type: 'text' as const,
          text: JSON.stringify(delegation, null, 2),
        }],
      };
    }
  );

  // =========================================================================
  // 10. iOS MODELS (Prompt Delegation to filesystem)
  // =========================================================================

  server.tool(
    'amber_ios_models',
    'List all Swift models in the AmberApp iOS project. ' +
      'Returns instructions for Claude to parse the Models/ directory structure.',
    {
      filter: z.string().optional().describe('Filter models by name pattern'),
    },
    async ({ filter }) => {
      const repoPath = AMBER_REPO_PATH;
      const delegation = {
        action: 'prompt_delegation',
        instructions: `Use filesystem tools to list all Swift models in the Amber iOS app:
1. Search for Swift model files in: ${repoPath ? repoPath + '/AmberApp' : 'the amber repo AmberApp/'}
2. Look in: Models/, Sources/Models/, AmberApp/Models/
3. Also search for files containing "struct.*: Codable" or "struct.*: Identifiable"
4. For each model file, extract: struct/class name, properties (name + type), protocol conformances
${filter ? `5. Filter to models matching "${filter}"` : '5. List all models'}
6. Format as a structured list grouped by directory`,
        context: {
          repo_path: repoPath,
          filter: filter || null,
          common_locations: [
            'AmberApp/Models/',
            'AmberApp/Sources/Models/',
            'ios/AmberApp/Models/',
          ],
        },
      };
      return {
        content: [{
          type: 'text' as const,
          text: JSON.stringify(delegation, null, 2),
        }],
      };
    }
  );

  // =========================================================================
  // 11. iOS VIEWS (Prompt Delegation to filesystem)
  // =========================================================================

  server.tool(
    'amber_ios_views',
    'List all SwiftUI views in the AmberApp iOS project. ' +
      'Returns instructions for Claude to parse the Views/ directory structure.',
    {
      filter: z.string().optional().describe('Filter views by name pattern'),
    },
    async ({ filter }) => {
      const repoPath = AMBER_REPO_PATH;
      const delegation = {
        action: 'prompt_delegation',
        instructions: `Use filesystem tools to list all SwiftUI views in the Amber iOS app:
1. Search for SwiftUI view files in: ${repoPath ? repoPath + '/AmberApp' : 'the amber repo AmberApp/'}
2. Look in: Views/, Sources/Views/, Screens/, Features/
3. Search for files containing "struct.*: View" (SwiftUI view protocol)
4. For each view file, extract: view name, any @State/@Binding/@ObservedObject properties, child views used
${filter ? `5. Filter to views matching "${filter}"` : '5. List all views'}
6. Format as a tree structure showing the view hierarchy by directory`,
        context: {
          repo_path: repoPath,
          filter: filter || null,
          common_locations: [
            'AmberApp/Views/',
            'AmberApp/Sources/Views/',
            'AmberApp/Screens/',
            'AmberApp/Features/',
            'ios/AmberApp/Views/',
          ],
        },
      };
      return {
        content: [{
          type: 'text' as const,
          text: JSON.stringify(delegation, null, 2),
        }],
      };
    }
  );

  // =========================================================================
  // 12. BACKEND SERVICES (Prompt Delegation to filesystem)
  // =========================================================================

  server.tool(
    'amber_backend_services',
    'List all backend services and their dependencies in the Amber Fastify backend. ' +
      'Returns instructions for Claude to analyze the service architecture.',
    {},
    async () => {
      const repoPath = AMBER_REPO_PATH;
      const delegation = {
        action: 'prompt_delegation',
        instructions: `Use filesystem tools to analyze the Amber backend service architecture:
1. Read the backend package.json from: ${repoPath ? repoPath + '/backend' : 'the amber repo backend/'}
2. List all service files in: src/services/, src/lib/, src/modules/
3. For each service, identify:
   - Service name and purpose (from file name and exports)
   - Dependencies (imports from other services)
   - External integrations (Privy, PostgreSQL, GCP, etc.)
4. Read the main app entry point (src/index.ts or src/app.ts) to understand plugin registration
5. Check for environment variables used (process.env references)
6. Format as: Service Name | Purpose | Dependencies | External Integrations`,
        context: {
          repo_path: repoPath,
          common_locations: [
            'backend/src/services/',
            'backend/src/lib/',
            'backend/src/modules/',
            'backend/src/plugins/',
          ],
          key_integrations: ['privy', 'postgresql', 'drizzle', 'gcp', 'cloud-run'],
        },
      };
      return {
        content: [{
          type: 'text' as const,
          text: JSON.stringify(delegation, null, 2),
        }],
      };
    }
  );

  // =========================================================================
  // 13. RUN TYPECHECK (Prompt Delegation)
  // =========================================================================

  server.tool(
    'amber_run_typecheck',
    'Run TypeScript type checking on the Amber backend. ' +
      'Returns instructions for Claude to execute tsc --noEmit and report results.',
    {
      fix_suggestions: z.boolean().default(false).describe('Include fix suggestions for each error'),
    },
    async ({ fix_suggestions }) => {
      const repoPath = AMBER_REPO_PATH;
      const delegation = {
        action: 'prompt_delegation',
        instructions: `Run TypeScript type checking on the Amber backend:
1. Navigate to: ${repoPath ? repoPath + '/backend' : 'the amber repo backend/'}
2. Run: npx tsc --noEmit 2>&1
3. Parse the output:
   - Count total errors and warnings
   - Group errors by file
   - Show each error with: file, line, column, error code, message
${fix_suggestions ? `4. For each error, suggest a fix based on the error code and context` : '4. Just list the errors without fix suggestions'}
5. If there are no errors, report "TypeScript compilation successful - no errors"
6. Summary format: X errors in Y files`,
        context: {
          repo_path: repoPath,
          fix_suggestions,
        },
      };
      return {
        content: [{
          type: 'text' as const,
          text: JSON.stringify(delegation, null, 2),
        }],
      };
    }
  );

  process.stderr.write(
    'Registered 13 tools: amber_health_check, amber_list_sprint_tasks, ' +
    'amber_get_task_status, amber_update_task, amber_list_api_routes, ' +
    'amber_db_schema, amber_deploy_status, amber_create_branch, ' +
    'amber_create_pr, amber_ios_models, amber_ios_views, ' +
    'amber_backend_services, amber_run_typecheck\n'
  );
}

// ---------------------------------------------------------------------------
// Server Creation
// ---------------------------------------------------------------------------

function createServer(): McpServer {
  const server = new McpServer({
    name: 'togari-amber',
    version: '1.0.0',
  });

  registerTools(server);

  return server;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

async function main(): Promise<void> {
  const port = process.env.PORT;

  if (port) {
    // HTTP mode — Railway deployment
    const app = express();

    // CORS
    app.use((_req, res, next) => {
      res.setHeader('Access-Control-Allow-Origin', '*');
      res.setHeader('Access-Control-Allow-Headers', 'Authorization, x-api-key, Content-Type, Mcp-Session-Id');
      res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
      next();
    });

    app.options('*', (_req, res) => {
      res.status(204).end();
    });

    // Health check
    app.get('/health', (_req, res) => {
      res.json({
        status: 'ok',
        service: 'togari-amber',
        version: '1.0.0',
        tools: 13,
      });
    });

    // SSE transport
    const transports = new Map<string, SSEServerTransport>();

    app.get('/sse', async (_req, res) => {
      const transport = new SSEServerTransport('/messages', res);
      const sessionId = transport.sessionId;
      transports.set(sessionId, transport);

      res.on('close', () => {
        transports.delete(sessionId);
      });

      const server = createServer();
      await server.connect(transport);
    });

    app.post('/messages', async (req, res) => {
      const sessionId = req.query.sessionId as string;
      const transport = transports.get(sessionId);
      if (!transport) {
        res.status(400).json({ error: 'Unknown session' });
        return;
      }
      await transport.handlePostMessage(req, res);
    });

    app.listen(Number(port), () => {
      process.stderr.write(
        `Togari Amber MCP Server listening on port ${port} (SSE mode)\n` +
        `Health: http://localhost:${port}/health\n`
      );
    });
  } else {
    // stdio mode — Claude Desktop
    const server = createServer();
    const transport = new StdioServerTransport();
    await server.connect(transport);
  }
}

main().catch((error) => {
  process.stderr.write(`Fatal error starting Togari Amber MCP Server: ${error}\n`);
  process.exit(1);
});
