import os

from taskweaver.plugin import Plugin, register_plugin

from project.sc_pipeline.sc_dag import DEFAULT_STORAGE_DIR
from project.sc_pipeline.sc_run import get_runner


@register_plugin
class SingleCellPipeline(Plugin):
    def __call__(self, target_stage: str, **kwargs):
        """Run the single-cell pipeline when TaskWeaver calls this plugin."""
        try:
            storage_root = self.config.get("storage_dir", DEFAULT_STORAGE_DIR)
            storage_dir = os.path.join(storage_root, self.ctx.session_id)
            runner = get_runner(ctx=self.ctx, config=self.config, storage_dir=storage_dir)
            return runner.execute(target_stage=target_stage, **kwargs)
        except Exception as exc:
            raise RuntimeError(f"Single-cell pipeline failed: {exc}") from exc
