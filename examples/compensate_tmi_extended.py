import matplotlib.pyplot as plt
from numpy.typing import ArrayLike
from sgl2020 import Sgl2020

from deinterf.compensator.tmi.linear import Terms, TollesLawson
from deinterf.foundation.sensors import MagVector, Tmi, DirectionalCosine
from deinterf.metrics.fom import improve_rate, noise_level
from deinterf.utils.data_ioc import DataIoC, DataNDArray
from deinterf.utils.transform import magvec2dircosine


class InertialAttitude(DataNDArray):
    def __new__(cls, yaw: ArrayLike, pitch: ArrayLike, roll: ArrayLike):
        return super().__new__(cls, yaw, pitch, roll)

    @property
    def yaw(self):
        return self[:, 0]

    @property
    def pitch(self):
        return self[:, 1]

    @property
    def roll(self):
        return self[:, 2]


class InsDirectionalCosine(DirectionalCosine):
    @classmethod
    def __build__(cls, container: DataIOC) -> DirectionalCosine:
        dir_cosine = magvec2dircosine(container[InertialAttitude])
        return cls(*dir_cosine.T)


if __name__ == "__main__":
    surv_d = (
        Sgl2020()
        .line(["1002.02"])
        .source(
            [
                "flux_b_x",
                "flux_b_y",
                "flux_b_z",
                "mag_3_uc",
                'ins_yaw',
                'ins_pitch',
                'ins_roll',
            ]
        )
        .take()
    )
    flt_d = surv_d["1002.02"]

    # 数据准备
    tmi_with_interf = Tmi(tmi=flt_d["mag_3_uc"])
    fom_data = DataIoC().with_data(
        MagVector[1](bx=flt_d["flux_b_x"], by=flt_d["flux_b_y"], bz=flt_d["flux_b_z"]),
        InertialAttitude[1](yaw=flt_d["ins_yaw"], pitch=flt_d["ins_pitch"], roll=flt_d["ins_roll"]),
    )
    fom_data.add_provider(DirectionalCosine, InsDirectionalCosine)

    # 创建补偿器
    compensator = TollesLawson(terms=Terms.Terms_16[1])
    compensator.fit(fom_data, tmi_with_interf)

    # 补偿给定信号
    tmi_clean = compensator.transform(fom_data, tmi_with_interf)

    # 或者一步到位，拟合与补偿自身
    tmi_clean = compensator.fit_transform(fom_data, tmi_with_interf)

    # 仅预测磁干扰
    interf = compensator.predict(fom_data)

    # 评估磁补偿性能
    comped_noise_level = noise_level(tmi_clean)
    print(f"{comped_noise_level=}")

    ir = improve_rate(tmi_with_interf, tmi_clean)
    print(f"{ir=}")

    # 简要绘图
    plt.plot(tmi_with_interf, label="tmi_with_interf")
    plt.plot(tmi_clean, label="tmi_clean")
    plt.legend()
    plt.show()
