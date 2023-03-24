#%%
from __init__ import *
from tqdm import tqdm
from student_model import StudentModel, STUDENT_MODEL, STUDENT_CONFIG
from teacher_model import TeacherModel, TEACHER_MODEL, TEACHER_CONFIG
from distillation_losses import TeacherStudentCELoss, TeacherStudentCosineEmbeddingLoss
#%%


loss_dict = {
    'teacher_student_ce': TeacherStudentCELoss,
    'teacher_student_layer_loss': TeacherStudentCosineEmbeddingLoss,
}
#%%
class DistillationLoss(nn.Module):
    def __init__(self, distillation_type='teacher_student_ce',alpha=0.5, beta=0.5, ) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.loss_fn = loss_dict[distillation_type]()
    def forward(self, student_input, teacher_input):
        return self.loss_fn(student_input, teacher_input)
#%%
class Distiller(nn.Module):
    def __init__(
        self, teacher_model, student_model, distillation_type=None, alpha=None, beta=None
    ):
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.distillation_type = distillation_type
        self.alpha = alpha
        self.beta = beta
        self.trainable_params = [p for p in student_model.parameters() if p.requires_grad==True]
        self.opt = torch.optim.AdamW(params = [p for p in student_model.parameters() if p.requires_grad==True], lr=0.001)
        self.loss_fn = DistillationLoss(distillation_type)
    def train(self, train_data, test_data=None, opt=None, lr=0.001, learning_scheduler=None, epochs=10):
        for epoch in range(epochs):
            self.train_one_epoch(train_data, test_data)
            
    def train_one_epoch(self, train_data, test_data):
        for data in tqdm(train_data):
            teacher_output = self.teacher_model(data)
            student_output = self.student_model(data)
            assert hasattr(student_output, 'logits') and hasattr(teacher_output, 'logits'), "Please design the models so that they follow the same output pattern as Huggingface if you wish to use the pre-defined losses"
            loss = self.loss_fn(student_output, teacher_output)
            loss.backward()
            self.opt.step()
            self.opt.zero_grad()
    
    def validate(self, test_data):
        c=0
        t=0
        for data in tqdm(test_data):
            teacher_output = self.teacher_model(data)
            student_output = self.student_model(data)
            if hasattr(teacher_output, 'logits') and hasattr(student_output, 'logits'):
                s_logits = student_output.logits
                t_logits = teacher_output.logits
                s_pred = torch.argmax(s_logits, dim=-1)
                t_pred = torch.argmax(t_logits, dim=-1)
                matches = s_pred==t_pred
                c+=torch.sum(matches)
                t+=len(matches)
        print("accuracy:", c/t)
        return c/t
            
DISTILLER = Distiller(TEACHER_MODEL, STUDENT_MODEL)
