from src.school import ArtSchool
from src.students.student_dcgan import StudentDCGAN
from src.tutors.tutor_dcgan import TutorDCGAN
from pytorch_lightning import Trainer


def main():
    img_shape = (3, 64, 64)

    artist_name = input("What is the name of the artist you want to create? ")
    art_type = input("What art type do you want {} to make? ".format(artist_name))
    n = int(input("How many images of {} do you want {} to train on? ".format(art_type, artist_name)))
    student = StudentDCGAN(
        latent_dim=100,
        img_shape=img_shape,
        art_type=art_type,
        name=artist_name,
        generator_features=150
    )
    tutor = TutorDCGAN(
        img_shape=img_shape,
        art_type=art_type,
        name="{}'s Tutor".format(artist_name),
        discriminator_features=50
    )

    school = ArtSchool(
        student=student,
        tutor=tutor,
        batch_size=32,
        lr=1e-3,
        n=n
    )
    art_program = Trainer(
        auto_select_gpus=True,
        max_epochs=10000,
        gpus=1
    )
    art_program.fit(school)


if __name__ == '__main__':
    main()
