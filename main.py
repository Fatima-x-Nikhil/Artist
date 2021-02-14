from templates.school import Student, Tutor, ArtSchool
from pytorch_lightning import Trainer


def main():
    img_shape = (3, 64, 64)
    student = Student(latent_dim=100, img_shape=img_shape, art_type="celeba", name="Ray")
    tutor = Tutor(img_shape=img_shape)
    school = ArtSchool(student, tutor, batch_size=1024)
    art_program = Trainer(auto_select_gpus=True, max_epochs=10000, gpus=1)
    result = art_program.fit(school)
    print(result)


if __name__ == '__main__':
    main()

