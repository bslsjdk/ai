package bslsjdk.mcnpu;

import org.json.JSONObject;

public final class AgentExecutor {
    public interface Observer {
        void onState(AgentTask task);
    }

    private final AgentToolRegistry registry;
    private final Observer observer;
    private final int maxSteps;

    public AgentExecutor(AgentToolRegistry registry, Observer observer, int maxSteps) {
        this.registry = registry;
        this.observer = observer;
        this.maxSteps = Math.max(1, maxSteps);
    }

    public JSONObject call(AgentTask task, String toolId, JSONObject args) {
        if (task == null) return error("missing_task");
        if (task.steps >= maxSteps) {
            task.state = AgentTask.State.FAILED;
            notifyState(task);
            return error("step_limit");
        }

        AgentTool tool = registry == null ? null : registry.get(toolId);
        if (tool == null) {
            task.state = AgentTask.State.FAILED;
            notifyState(task);
            return error("unknown_tool");
        }

        if (tool.permission() == AgentPermission.BLOCK) {
            task.state = AgentTask.State.FAILED;
            notifyState(task);
            return error("tool_blocked");
        }

        task.state = AgentTask.State.TOOL_CALL;
        task.steps++;
        notifyState(task);

        try {
            JSONObject result = tool.execute(args == null ? new JSONObject() : args);
            task.state = AgentTask.State.OBSERVING;
            notifyState(task);
            return result == null ? new JSONObject() : result;
        } catch (Throwable t) {
            task.state = AgentTask.State.FAILED;
            notifyState(task);
            return error(t.getClass().getSimpleName());
        }
    }

    private JSONObject error(String reason) {
        JSONObject out = new JSONObject();
        try {
            out.put("ok", false);
            out.put("error", reason);
        } catch (Throwable ignored) {}
        return out;
    }

    private void notifyState(AgentTask task) {
        if (observer != null) observer.onState(task);
    }
}
