from fastapi import FastAPI

from nebula3_experts.service.base_expert import BaseExpert
from nebula3_experts.app import ExpertApp
from nebula3_experts.common.models import ExpertParam

class TrackerExpert(BaseExpert):
    def __init__(self):
        super().__init__()
        # after init all
        self.set_active()
    def get_name(self):
        return "TrackerExpert"

    def add_expert_apis(self, app: FastAPI):
        pass
        # @app.get("/my-expert")
        # def get_my_expert(q: Optional[str] = None):
        #     return {"expert": "my-expert" }

    def predict(self, expert_params: ExpertParam):
        """ handle new movie """
        movie = self.movie_db.get_movie(expert_params.movie_id)
        print(f'Predicting movie: {expert_params.movie_id}')
        return { 'result': { 'movie_id' : expert_params.movie_id, 'info': movie , 'extra_params': expert_params.extra_params} }


tracker_expert = TrackerExpert()
expert_app = ExpertApp(expert=tracker_expert)
app = expert_app.get_app()
expert_app.run()