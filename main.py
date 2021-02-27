from src.school import ArtSchool
from src.students.student_dcgan import StudentDCGAN
from src.tutors.tutor_dcgan import TutorDCGAN
from pytorch_lightning import Trainer


def main():
    img_shape = (3, 64, 64)

    student = StudentDCGAN(
        latent_dim=100,
        img_shape=img_shape,
        art_type="celeba",
        name="Ray",
        generator_features=128
    )
    tutor = TutorDCGAN(
        img_shape=img_shape,
        art_type="celeba",
        name="Ray's Tutor",
        discriminator_features=32
    )

    school = ArtSchool(
        student=student,
        tutor=tutor,
        batch_size=1024,
        lr=1e-3,
    )
    art_program = Trainer(
        auto_select_gpus=True,
        max_epochs=10000,
        gpus=1
    )
    art_program.fit(school)


if __name__ == '__main__':
    main()

